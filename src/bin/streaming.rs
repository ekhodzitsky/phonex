//! Streaming transcription CLI — feed a WAV file to StreamingPipeline chunk by chunk.

use clap::{Parser, ValueEnum};

use phonex::audio::AudioPreprocessor;
use phonex::model_config::ModelInfo;
use phonex::streaming_pipeline::StreamingPipeline;

const TARGET_SAMPLE_RATE: u32 = 16000;

#[derive(Parser, Debug)]
#[command(name = "streaming")]
#[command(about = "Stream-transcribe a WAV file using Sherpa-ONNX Zipformer")]
struct Args {
    /// Directory containing encoder.onnx, decoder.onnx, joiner.onnx, tokenizer.model
    #[arg(long)]
    model_dir: Option<String>,

    /// Language model to use (streaming models only)
    #[arg(long, value_enum)]
    language: Option<Language>,

    /// Path to the input WAV file
    #[arg(long, short)]
    wav: String,

    /// Chunk size in milliseconds
    #[arg(long, default_value_t = 500)]
    chunk_ms: usize,

    /// Optional path to silero_vad.onnx
    #[arg(long)]
    vad: Option<String>,
}

#[derive(Clone, Debug, ValueEnum)]
enum Language {
    /// English streaming (LibriSpeech + GigaSpeech)
    En20230621,
    /// English streaming (LibriSpeech)
    En20230626,
    /// French streaming
    Fr20230414,
    /// German streaming (Kroko)
    DeKroko20250806,
    /// Spanish streaming (Kroko)
    EsKroko20250806,
    /// Korean streaming
    Ko20240616,
}

impl Language {
    fn model_dir(&self) -> &'static str {
        match self {
            Language::En20230621 => "models/sherpa-onnx-streaming-zipformer-en-2023-06-21",
            Language::En20230626 => "models/sherpa-onnx-streaming-zipformer-en-2023-06-26",
            Language::Fr20230414 => "models/sherpa-onnx-streaming-zipformer-fr-2023-04-14",
            Language::DeKroko20250806 => "models/sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06",
            Language::EsKroko20250806 => "models/sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06",
            Language::Ko20240616 => "models/sherpa-onnx-streaming-zipformer-korean-2024-06-16",
        }
    }
}

fn resolve_model_dir(model_dir: Option<String>, language: Option<Language>) -> String {
    model_dir.unwrap_or_else(|| {
        language
            .map(|l| l.model_dir().to_string())
            .unwrap_or_else(|| "models/sherpa-onnx-streaming-zipformer-en-2023-06-21".to_string())
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model_dir = resolve_model_dir(args.model_dir, args.language);

    eprintln!("Loading model from: {}", model_dir);
    phonex::model::ensure_model(&model_dir)?;

    let vad_path = args.vad.or_else(|| {
        let default = "models/silero_vad.onnx";
        if std::path::Path::new(default).exists() {
            Some(default.to_string())
        } else {
            None
        }
    });

    if let Some(ref path) = vad_path {
        eprintln!("VAD enabled: {}", path);
    } else {
        eprintln!("VAD disabled");
    }

    let info = ModelInfo::from_model_dir(&model_dir)?;
    let mut pipeline =
        StreamingPipeline::from_model_dir(&model_dir, &info, vad_path.as_deref())?;

    eprintln!("Reading WAV: {}", args.wav);
    let (samples, sample_rate) = AudioPreprocessor::read_wav(&args.wav)?;
    eprintln!(
        "Original sample rate: {} Hz, samples: {}",
        sample_rate,
        samples.len()
    );

    let samples = if sample_rate as u32 != TARGET_SAMPLE_RATE {
        eprintln!("Resampling {} Hz → {} Hz", sample_rate, TARGET_SAMPLE_RATE);
        phonex::inference::resample(&samples, sample_rate as u32, TARGET_SAMPLE_RATE)?
    } else {
        samples
    };

    let chunk_samples = TARGET_SAMPLE_RATE as usize * args.chunk_ms / 1000;
    eprintln!(
        "Streaming with chunk size: {} ms ({} samples)",
        args.chunk_ms, chunk_samples
    );

    for chunk in samples.chunks(chunk_samples) {
        let _tokens = pipeline.accept_audio(chunk)?;
        if let Some(text) = pipeline.take_final_text() {
            println!("{}", text);
        }
    }

    let text = pipeline.flush()?;
    println!("{}", text);

    Ok(())
}
