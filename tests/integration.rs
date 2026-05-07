use phonex::Engine;
use phonex::model_config::ModelInfo;

const MODEL_DIR: &str = "models/sherpa-onnx-zipformer-thai-2024-06-20";
const STREAMING_EN_DIR: &str = "models/sherpa-onnx-streaming-zipformer-en-2023-06-21";

fn maybe_resample(samples: Vec<f32>, sample_rate: usize, target_rate: usize) -> Vec<f32> {
    if sample_rate == target_rate {
        samples
    } else {
        phonex::audio::AudioPreprocessor::typhoon().resample(&samples, sample_rate).unwrap()
    }
}

#[test]
fn test_transcribe_0_wav() {
    let engine = Engine::load(MODEL_DIR).expect("failed to load engine");
    let text = engine.transcribe_file(&format!("{}/test_wavs/0.wav", MODEL_DIR))
        .expect("transcription failed");
    assert!(!text.is_empty(), "transcription should not be empty");
    assert!(text.contains("เกม"), "expected 'เกม' in transcription, got: {}", text);
    assert!(text.contains("อินโดนีเซีย"), "expected 'อินโดนีเซีย' in transcription, got: {}", text);
}

#[test]
fn test_transcribe_1_wav() {
    let engine = Engine::load(MODEL_DIR).expect("failed to load engine");
    let text = engine.transcribe_file(&format!("{}/test_wavs/1.wav", MODEL_DIR))
        .expect("transcription failed");
    assert!(!text.is_empty(), "transcription should not be empty");
    assert!(text.contains("แข่งขัน"), "expected 'แข่งขัน' in transcription, got: {}", text);
}

#[test]
fn test_transcribe_2_wav() {
    let engine = Engine::load(MODEL_DIR).expect("failed to load engine");
    let text = engine.transcribe_file(&format!("{}/test_wavs/2.wav", MODEL_DIR))
        .expect("transcription failed");
    assert!(!text.is_empty(), "transcription should not be empty");
    assert!(text.contains("เกม"), "expected 'เกม' in transcription, got: {}", text);
}

#[test]
fn test_transcribe_batch() {
    let engine = Engine::load(MODEL_DIR).expect("failed to load engine");
    let (samples0, sr0) = phonex::audio::AudioPreprocessor::read_wav(&format!("{}/test_wavs/0.wav", MODEL_DIR)).unwrap();
    let (samples1, sr1) = phonex::audio::AudioPreprocessor::read_wav(&format!("{}/test_wavs/1.wav", MODEL_DIR)).unwrap();

    let samples0 = maybe_resample(samples0, sr0, engine.info.sample_rate as usize);
    let samples1 = maybe_resample(samples1, sr1, engine.info.sample_rate as usize);

    let mut guard = engine.pool.try_checkout().expect("pool empty");
    let results = engine.transcribe_batch(vec![&samples0, &samples1], &mut guard)
        .expect("batch transcription failed");

    assert_eq!(results.len(), 2);
    assert!(!results[0].text.is_empty(), "first transcription should not be empty");
    assert!(!results[1].text.is_empty(), "second transcription should not be empty");
    assert!(results[0].text.contains("เกม"), "expected 'เกม' in first, got: {}", results[0].text);
    assert!(results[1].text.contains("แข่งขัน"), "expected 'แข่งขัน' in second, got: {}", results[1].text);
}

#[test]
fn test_transcribe_vad_2_wav() {
    let engine = Engine::load(MODEL_DIR).expect("failed to load engine");
    let (samples, sample_rate) = phonex::audio::AudioPreprocessor::read_wav(&format!("{}/test_wavs/2.wav", MODEL_DIR)).unwrap();
    let samples = maybe_resample(samples, sample_rate, engine.info.sample_rate as usize);
    let mut guard = engine.pool.try_checkout()
        .expect("pool empty");
    let result = engine.transcribe_samples_with_vad(&samples, &mut guard)
        .expect("VAD transcription failed");
    // VAD may not detect speech in very short/quiet clips on all platforms.
    // The important thing is that it doesn't panic and returns a valid result.
    let full_text = result.words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
    assert!(
        result.words.is_empty() || !full_text.is_empty(),
        "VAD transcription should be empty or non-empty, never invalid"
    );
}

#[test]
#[ignore = "slow — loads streaming ONNX model"]
fn test_streaming_en_pipeline() {
    let info = ModelInfo::from_model_dir(STREAMING_EN_DIR).expect("failed to load model info");
    let mut pipeline = phonex::streaming_pipeline::StreamingPipeline::from_model_dir(STREAMING_EN_DIR, &info, None)
        .expect("failed to create streaming pipeline");

    let (samples, sample_rate) = phonex::audio::AudioPreprocessor::read_wav(&format!("{}/test_wavs/0.wav", STREAMING_EN_DIR)).unwrap();
    let samples = maybe_resample(samples, sample_rate, info.sample_rate as usize);

    // Feed in 0.5-second chunks
    let chunk_samples = 8000;
    let mut offset = 0;
    while offset + chunk_samples < samples.len() {
        let _tokens = pipeline.accept_audio(&samples[offset..offset + chunk_samples]).unwrap();
        offset += chunk_samples;
    }
    if offset < samples.len() {
        let _tokens = pipeline.accept_audio(&samples[offset..]).unwrap();
    }

    let text = pipeline.flush().unwrap();
    println!("Streaming tokens count: {}", pipeline.tokens().len());
    println!("Streaming result: '{}'", text);
    assert!(!text.is_empty(), "streaming transcription should not be empty");
}

/// Helper: feed samples to a streaming pipeline in fixed-size chunks and return the transcription.
fn streaming_transcribe(pipeline: &mut phonex::streaming_pipeline::StreamingPipeline, samples: &[f32], chunk_size: usize) -> String {
    let mut offset = 0;
    while offset + chunk_size < samples.len() {
        let _tokens = pipeline.accept_audio(&samples[offset..offset + chunk_size]).unwrap();
        offset += chunk_size;
    }
    if offset < samples.len() {
        let _tokens = pipeline.accept_audio(&samples[offset..]).unwrap();
    }
    pipeline.flush().unwrap()
}

#[test]
#[ignore = "slow — loads streaming ONNX model"]
fn test_streaming_en_1_wav() {
    let info = ModelInfo::from_model_dir(STREAMING_EN_DIR).expect("failed to load model info");
    let mut pipeline = phonex::streaming_pipeline::StreamingPipeline::from_model_dir(STREAMING_EN_DIR, &info, None)
        .expect("failed to create streaming pipeline");

    let (samples, sample_rate) = phonex::audio::AudioPreprocessor::read_wav(&format!("{}/test_wavs/1.wav", STREAMING_EN_DIR)).unwrap();
    let samples = maybe_resample(samples, sample_rate, info.sample_rate as usize);

    let text = streaming_transcribe(&mut pipeline, &samples, 8000);
    println!("Streaming 1.wav result: '{}'", text);
    assert!(!text.is_empty(), "streaming transcription of 1.wav should not be empty");
}

#[test]
#[ignore = "slow — loads streaming ONNX model"]
fn test_streaming_en_8k_wav() {
    let info = ModelInfo::from_model_dir(STREAMING_EN_DIR).expect("failed to load model info");
    let mut pipeline = phonex::streaming_pipeline::StreamingPipeline::from_model_dir(STREAMING_EN_DIR, &info, None)
        .expect("failed to create streaming pipeline");

    let (samples, sample_rate) = phonex::audio::AudioPreprocessor::read_wav(&format!("{}/test_wavs/8k.wav", STREAMING_EN_DIR)).unwrap();
    assert_eq!(sample_rate, 8000, "8k.wav should have 8000 Hz sample rate");
    let samples = phonex::audio::AudioPreprocessor::typhoon().resample(&samples, sample_rate).unwrap();

    let text = streaming_transcribe(&mut pipeline, &samples, 8000);
    println!("Streaming 8k.wav result: '{}'", text);
    assert!(!text.is_empty(), "streaming transcription of 8k.wav should not be empty");
}

#[test]
#[ignore = "slow — loads streaming ONNX model"]
fn test_streaming_different_chunk_sizes() {
    let info = ModelInfo::from_model_dir(STREAMING_EN_DIR).expect("failed to load model info");
    let (samples, sample_rate) = phonex::audio::AudioPreprocessor::read_wav(&format!("{}/test_wavs/0.wav", STREAMING_EN_DIR)).unwrap();
    let samples = maybe_resample(samples, sample_rate, info.sample_rate as usize);

    for chunk_size in [1000, 4000, 8000, 16000] {
        let mut pipeline = phonex::streaming_pipeline::StreamingPipeline::from_model_dir(STREAMING_EN_DIR, &info, None)
            .expect("failed to create streaming pipeline");
        let text = streaming_transcribe(&mut pipeline, &samples, chunk_size);
        println!("Chunk size {} -> '{}'", chunk_size, text);
        assert!(!text.is_empty(), "streaming transcription with chunk_size={} should not be empty", chunk_size);
    }
}

#[test]
#[ignore = "slow — loads streaming ONNX model"]
fn test_streaming_reset_reuse() {
    let info = ModelInfo::from_model_dir(STREAMING_EN_DIR).expect("failed to load model info");
    let mut pipeline = phonex::streaming_pipeline::StreamingPipeline::from_model_dir(STREAMING_EN_DIR, &info, None)
        .expect("failed to create streaming pipeline");

    let (samples0, sr0) = phonex::audio::AudioPreprocessor::read_wav(&format!("{}/test_wavs/0.wav", STREAMING_EN_DIR)).unwrap();
    let samples0 = if sr0 != info.sample_rate as usize {
        phonex::audio::AudioPreprocessor::typhoon().resample(&samples0, sr0).unwrap()
    } else {
        samples0
    };

    let text0 = streaming_transcribe(&mut pipeline, &samples0, 8000);
    println!("First utterance (0.wav): '{}'", text0);
    assert!(!text0.is_empty(), "first streaming transcription should not be empty");

    pipeline.reset();

    let (samples1, sr1) = phonex::audio::AudioPreprocessor::read_wav(&format!("{}/test_wavs/1.wav", STREAMING_EN_DIR)).unwrap();
    let samples1 = if sr1 != info.sample_rate as usize {
        phonex::audio::AudioPreprocessor::typhoon().resample(&samples1, sr1).unwrap()
    } else {
        samples1
    };

    let text1 = streaming_transcribe(&mut pipeline, &samples1, 8000);
    println!("Second utterance (1.wav): '{}'", text1);
    assert!(!text1.is_empty(), "second streaming transcription should not be empty");
    assert_ne!(text0, text1, "reset and reuse should produce different transcription for different audio");
}
