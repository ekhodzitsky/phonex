//! phonex CLI — transcribe audio files or run the HTTP server.

use clap::{Parser, Subcommand, ValueEnum};
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

        /// Model directory (overrides --language)
        #[arg(short, long)]
        model_dir: Option<String>,

        /// Language model to use
        #[arg(short, long, value_enum)]
        language: Option<Language>,

        /// Output format
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Enable speaker diarization (requires diarization feature and model)
        #[arg(long)]
        diarize: bool,
    },

    /// Run the HTTP server.
    Serve {
        /// Address to bind to
        #[arg(short, long, default_value = "127.0.0.1")]
        bind: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Model directory (overrides --language)
        #[arg(short, long)]
        model_dir: Option<String>,

        /// Language model to use
        #[arg(short, long, value_enum)]
        language: Option<Language>,

        /// Number of parallel inference sessions
        #[arg(long, default_value_t = 1)]
        pool_size: usize,

        /// Path to speaker embedding ONNX model for diarization
        #[arg(long)]
        diarization_model: Option<String>,
    },
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Transcribe {
            file,
            model_dir,
            language,
            format,
            diarize,
        } => {
            let model_dir = resolve_model_dir(model_dir, language);
            phonex::model::ensure_model(&model_dir)?;
            let engine = phonex::Engine::load(&model_dir)?;
            #[cfg(feature = "diarization")]
            let engine = if diarize {
                engine.with_diarization("models/wespeaker_resnet34.onnx")
            } else {
                engine
            };
            #[cfg(not(feature = "diarization"))]
            let _ = diarize;
            match format.as_str() {
                "json" => {
                    #[cfg(feature = "diarization")]
                    let result = if diarize {
                        let (samples, sample_rate) = phonex::audio::AudioPreprocessor::read_wav(&file)?;
                        let samples = if sample_rate == engine.info.sample_rate as usize {
                            samples
                        } else {
                            phonex::audio::AudioPreprocessor::typhoon().resample(&samples, sample_rate)
                        };
                        let mut triplet = phonex::inference::SessionTriplet::from_model_dir(&model_dir, &engine.info)?;
                        engine.transcribe_samples_with_diarization(&samples, &mut triplet)?
                    } else {
                        engine.transcribe_file_with_details(&file)?
                    };
                    #[cfg(not(feature = "diarization"))]
                    let result = engine.transcribe_file_with_details(&file)?;
                    println!("{}", serde_json::to_string(&result)?);
                }
                _ => {
                    let text = engine.transcribe_file(&file)?;
                    println!("{}", text);
                }
            }
        }
        Commands::Serve {
            bind,
            port,
            model_dir,
            language,
            pool_size,
            diarization_model,
        } => {
            let model_dir = resolve_model_dir(model_dir, language);
            phonex::model::ensure_model(&model_dir)?;
            run_server(&bind, port, &model_dir, pool_size, diarization_model.as_deref())?;
        }
    }

    Ok(())
}

#[cfg(feature = "server")]
#[allow(unused_variables)]
fn run_server(
    bind: &str,
    port: u16,
    model_dir: &str,
    pool_size: usize,
    diarization_model: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use phonex::inference::pool::{SessionPool, SessionTriplet};
    use phonex::inference::Engine;
    use phonex::model_config::ModelInfo;
    use phonex::server;
    use phonex::tokenizer::Tokenizer;
    use std::net::SocketAddr;
    use std::sync::Arc;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let info = ModelInfo::from_model_dir(model_dir)?;
        let paths = phonex::model_config::discover_model_files(model_dir)?;

        let tokenizer = Arc::new(Tokenizer::from_file(
            paths.tokenizer.to_str().unwrap_or(""),
            paths.tokens.to_str().unwrap_or(""),
            info.blank_id,
        )?);

        let mut triplets = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            tracing::debug!(slot = i, "Loading ONNX sessions");
            triplets.push(SessionTriplet::from_model_dir(model_dir, &info)?);
        }

        let pool = SessionPool::new(triplets);
        let engine = Engine::new(pool, tokenizer, info.clone()).with_vad(
            paths
                .vad
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default()
                .as_str(),
        );
        #[cfg(feature = "diarization")]
        let engine = if let Some(model) = diarization_model {
            engine.with_diarization(model)
        } else {
            engine
        };
        let engine = Arc::new(engine);

        let addr: SocketAddr = format!("{}:{}", bind, port).parse()?;
        tracing::info!(%addr, "Starting server");

        let shutdown = tokio_util::sync::CancellationToken::new();
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let app = server::app_with_limits(
            engine,
            model_dir.to_string(),
            info,
            server::RuntimeLimits::default(),
            shutdown.clone(),
        );

        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                phonex::server::shutdown_signal().await;
                shutdown.cancel();
            })
            .await?;

        tracing::info!("Server shut down gracefully");
        Ok(())
    })
}

#[cfg(not(feature = "server"))]
fn run_server(
    _bind: &str,
    _port: u16,
    _model_dir: &str,
    _pool_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Server feature is not enabled. Rebuild with --features server");
    std::process::exit(1);
}


