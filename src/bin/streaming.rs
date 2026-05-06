//! Streaming transcription CLI — feed a WAV file to StreamingPipeline chunk by chunk.

use clap::Parser;

use phonex::audio::AudioPreprocessor;
use phonex::model_config::ModelInfo;
use phonex::streaming_pipeline::StreamingPipeline;

const TARGET_SAMPLE_RATE: u32 = 16000;

#[derive(Parser, Debug)]
#[command(name = "streaming")]
#[command(about = "Stream-transcribe a WAV file using Sherpa-ONNX Zipformer")]
struct Args {
    /// Directory containing encoder.onnx, decoder.onnx, joiner.onnx, tokenizer.model
    #[arg(long, default_value = "models/sherpa-onnx-streaming-zipformer-en-2023-06-21")]
    model_dir: String,

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("Loading model from: {}", args.model_dir);
    phonex::model::ensure_model(&args.model_dir)?;

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

    let info = ModelInfo::from_model_dir(&args.model_dir)?;
    let mut pipeline = StreamingPipeline::from_model_dir(&args.model_dir, &info, vad_path.as_deref())?;

    eprintln!("Reading WAV: {}", args.wav);
    let (samples, sample_rate) = AudioPreprocessor::read_wav(&args.wav)?;
    eprintln!("Original sample rate: {} Hz, samples: {}", sample_rate, samples.len());

    let samples = if sample_rate as u32 != TARGET_SAMPLE_RATE {
        eprintln!("Resampling {} Hz → {} Hz", sample_rate, TARGET_SAMPLE_RATE);
        phonex::inference::resample(&samples, sample_rate as u32, TARGET_SAMPLE_RATE)?
    } else {
        samples
    };

    let chunk_samples = TARGET_SAMPLE_RATE as usize * args.chunk_ms / 1000;
    eprintln!("Streaming with chunk size: {} ms ({} samples)", args.chunk_ms, chunk_samples);

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
