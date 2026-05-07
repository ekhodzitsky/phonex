//! phonex server — REST API + WebSocket for speech-to-text.

use std::net::SocketAddr;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use tracing_subscriber::EnvFilter;

use phonex::inference::pool::{SessionPool, SessionTriplet};
use phonex::inference::Engine;
use phonex::model_config::ModelInfo;
use phonex::server;
use phonex::server::RuntimeLimits;
use phonex::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "phonex-server")]
#[command(about = "Generic on-device speech-to-text server powered by Sherpa-ONNX Zipformer")]
struct Args {
    /// Directory containing encoder.onnx, decoder.onnx, joiner.onnx, tokenizer.model
    #[arg(long)]
    model_dir: Option<String>,

    /// Language model to use
    #[arg(long, value_enum)]
    language: Option<Language>,

    /// Address to bind to
    #[arg(short, long, default_value = "127.0.0.1")]
    bind: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Number of parallel inference sessions (pool size)
    #[arg(long, default_value_t = 1)]
    pool_size: usize,

    /// Optional API key for authentication (also set via PHONEX_API_KEY env var)
    #[arg(long, env = "PHONEX_API_KEY")]
    api_key: Option<String>,

    /// Comma-separated list of allowed CORS origins
    #[arg(long, value_delimiter = ',')]
    cors_origins: Option<Vec<String>>,

    /// Enable gRPC API on the given port (requires `grpc` feature)
    #[cfg(feature = "grpc")]
    #[arg(long)]
    grpc_port: Option<u16>,
}

#[derive(Clone, Debug, ValueEnum)]
enum Language {
    /// Cantonese (offline)
    Cantonese,
    /// Chinese + English bilingual (offline)
    Chinese,
    /// English (offline, LibriSpeech)
    English,
    /// Japanese (offline, ReazonSpeech)
    Japanese,
    /// Korean (offline)
    Korean,
    /// Russian — small model (offline)
    Russian,
    /// Thai (offline)
    Thai,
    /// Vietnamese — small int8 model (offline)
    Vietnamese,
}

impl Language {
    fn model_dir(&self) -> &'static str {
        match self {
            Language::Cantonese => "models/sherpa-onnx-zipformer-cantonese-2024-03-13",
            Language::Chinese => "models/sherpa-onnx-zipformer-zh-en-2023-11-22",
            Language::English => "models/sherpa-onnx-zipformer-en-2023-06-26",
            Language::Japanese => "models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
            Language::Korean => "models/sherpa-onnx-zipformer-korean-2024-06-24",
            Language::Russian => "models/sherpa-onnx-small-zipformer-ru-2024-09-18",
            Language::Thai => "models/sherpa-onnx-zipformer-thai-2024-06-20",
            Language::Vietnamese => "models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        }
    }
}

fn resolve_model_dir(model_dir: Option<String>, language: Option<Language>) -> String {
    model_dir.unwrap_or_else(|| {
        language
            .map(|l| l.model_dir().to_string())
            .unwrap_or_else(|| Language::English.model_dir().to_string())
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let model_dir = resolve_model_dir(args.model_dir, args.language);

    tracing::info!(model_dir = %model_dir, pool_size = args.pool_size, "Loading model");
    phonex::model::ensure_model(&model_dir)?;

    let info = ModelInfo::from_model_dir(&model_dir)?;
    let paths = phonex::model_config::discover_model_files(&model_dir)?;

    let tokenizer = Arc::new(Tokenizer::from_file(
        paths.tokenizer.to_str().unwrap_or(""),
        paths.tokens.to_str().unwrap_or(""),
        info.blank_id,
    )?);

    let mut triplets = Vec::with_capacity(args.pool_size);
    for i in 0..args.pool_size {
        tracing::debug!(slot = i, "Loading ONNX sessions");
        triplets.push(SessionTriplet::from_model_dir(&model_dir, &info)?);
    }

    let pool = SessionPool::new(triplets);
    let engine = Arc::new(
        Engine::new(pool, tokenizer, info.clone()).with_vad(
            paths
                .vad
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default()
                .as_str(),
        ),
    );

    let addr: SocketAddr = format!("{}:{}", args.bind, args.port).parse()?;
    tracing::info!(%addr, "Starting HTTP server");

    let mut limits = RuntimeLimits {
        api_key: args.api_key,
        ..RuntimeLimits::default()
    };
    if let Some(origins) = args.cors_origins {
        limits.cors_origins = origins;
    }

    let shutdown = tokio_util::sync::CancellationToken::new();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let app = server::app_with_limits(engine.clone(), model_dir.clone(), info.clone(), limits, shutdown.clone());

    let shutdown_http = shutdown.clone();
    let http_handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown_http.cancelled().await;
            })
            .await
    });

    #[cfg(feature = "grpc")]
    let grpc_handle = if let Some(grpc_port) = args.grpc_port {
        let grpc_addr: SocketAddr = format!("{}:{}", args.bind, grpc_port).parse()?;
        tracing::info!(%grpc_addr, "Starting gRPC server");
        let grpc_svc = server::grpc::PhonexGrpcService::new(engine, info, model_dir);
        Some(tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(grpc_svc.into_server())
                .serve(grpc_addr)
                .await
        }))
    } else {
        None
    };

    shutdown_signal().await;
    shutdown.cancel();

    if let Err(e) = http_handle.await {
        tracing::error!("HTTP server task failed: {e}");
    }

    #[cfg(feature = "grpc")]
    if let Some(h) = grpc_handle {
        if let Err(e) = h.await {
            tracing::error!("gRPC server task failed: {e}");
        }
    }

    tracing::info!("Server shut down gracefully");
    Ok(())
}

async fn shutdown_signal() {
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("Failed to install SIGTERM handler");
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .expect("Failed to install SIGINT handler");

    tokio::select! {
        _ = sigterm.recv() => tracing::info!("Received SIGTERM"),
        _ = sigint.recv() => tracing::info!("Received SIGINT"),
    }
}
