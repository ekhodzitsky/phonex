//! phonex CLI — transcribe audio files or run the HTTP server.

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "phonex")]
#[command(about = "Generic on-device speech-to-text powered by Sherpa-ONNX Zipformer")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Transcribe an audio file to text.
    Transcribe {
        /// Path to audio file (wav, mp3, ogg, flac, aac, etc.)
        file: String,

        /// Model directory
        #[arg(short, long, default_value = "models/sherpa-onnx-zipformer-thai-2024-06-20")]
        model_dir: String,

        /// Output format
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Run the HTTP server.
    Serve {
        /// Address to bind to
        #[arg(short, long, default_value = "127.0.0.1")]
        bind: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Model directory
        #[arg(short, long, default_value = "models/sherpa-onnx-zipformer-thai-2024-06-20")]
        model_dir: String,

        /// Number of parallel inference sessions
        #[arg(long, default_value_t = 1)]
        pool_size: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Transcribe { file, model_dir, format } => {
            phonex::model::ensure_model(&model_dir)?;
            let engine = phonex::Engine::load(&model_dir)?;
            match format.as_str() {
                "json" => {
                    let result = engine.transcribe_file_with_details(&file)?;
                    println!("{}", serde_json::to_string(&result)?);
                }
                _ => {
                    let text = engine.transcribe_file(&file)?;
                    println!("{}", text);
                }
            }
        }
        Commands::Serve { bind, port, model_dir, pool_size } => {
            phonex::model::ensure_model(&model_dir)?;
            run_server(&bind, port, &model_dir, pool_size)?;
        }
    }

    Ok(())
}

#[cfg(feature = "server")]
fn run_server(bind: &str, port: u16, model_dir: &str, pool_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use phonex::tokenizer::Tokenizer;
    use phonex::model_config::ModelInfo;
    use phonex::inference::pool::{SessionPool, SessionTriplet};
    use phonex::inference::Engine;
    use phonex::server;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let info = ModelInfo::from_model_dir(model_dir)?;
        let paths = phonex::model_config::discover_model_files(model_dir)?;

        let tokenizer = Arc::new(
            Tokenizer::from_file(
                paths.tokenizer.to_str().unwrap_or(""),
                paths.tokens.to_str().unwrap_or(""),
                info.blank_id,
            )?,
        );

        let mut triplets = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            tracing::debug!(slot = i, "Loading ONNX sessions");
            triplets.push(SessionTriplet::from_model_dir(model_dir, &info)?);
        }

        let pool = SessionPool::new(triplets);
        let engine = Arc::new(Engine::new(pool, tokenizer, info.clone())
            .with_vad(paths.vad.map(|p| p.to_string_lossy().to_string()).unwrap_or_default().as_str()));

        let addr: SocketAddr = format!("{}:{}", bind, port).parse()?;
        tracing::info!(%addr, "Starting server");

        let shutdown = tokio_util::sync::CancellationToken::new();
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let app = server::app_with_limits(engine, model_dir.to_string(), info, server::RuntimeLimits::default(), shutdown.clone());

        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown_signal().await;
                shutdown.cancel();
            })
            .await?;

        tracing::info!("Server shut down gracefully");
        Ok(())
    })
}

#[cfg(not(feature = "server"))]
fn run_server(_bind: &str, _port: u16, _model_dir: &str, _pool_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Server feature is not enabled. Rebuild with --features server");
    std::process::exit(1);
}

#[cfg(feature = "server")]
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
