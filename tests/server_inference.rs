//! Integration tests for the HTTP server with real inference.
//!
//! These tests load the real ONNX model and are therefore slower.
//! Run with: cargo test --test server_inference -- --ignored

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use phonex::encoder::OfflineEncoder;
use phonex::decoder::SherpaDecoder;
use phonex::joiner::SherpaJoiner;
use phonex::tokenizer::Tokenizer;
use phonex::inference::pool::{SessionPool, SessionTriplet};
use phonex::inference::Engine;
use std::sync::Arc;

const MODEL_DIR: &str = "models/sherpa-onnx-zipformer-thai-2024-06-20";

fn real_engine() -> (Arc<Engine>, phonex::model_config::ModelInfo) {
    let paths = phonex::model_config::discover_model_files(MODEL_DIR).unwrap();
    let tokenizer = Arc::new(
        Tokenizer::from_file(
            paths.tokenizer.to_str().unwrap_or(""),
            paths.tokens.to_str().unwrap_or(""),
            0,
        )
        .unwrap(),
    );

    let info = phonex::model_config::ModelInfo::from_model_dir(MODEL_DIR).unwrap();

    let encoder = OfflineEncoder::new(&format!("{}/encoder-epoch-12-avg-5.int8.onnx", MODEL_DIR), &info).unwrap();
    let decoder = SherpaDecoder::new(&format!("{}/decoder-epoch-12-avg-5.int8.onnx", MODEL_DIR), &info).unwrap();
    let joiner = SherpaJoiner::new(&format!("{}/joiner-epoch-12-avg-5.int8.onnx", MODEL_DIR), &info).unwrap();

    let pool = SessionPool::new(vec![SessionTriplet::new(encoder, decoder, joiner)]);
    (Arc::new(Engine::new(pool, tokenizer, info.clone())), info)
}

fn maybe_resample(samples: Vec<f32>, sample_rate: usize, target_rate: usize) -> Vec<f32> {
    if sample_rate == target_rate {
        samples
    } else {
        phonex::audio::AudioPreprocessor::typhoon().resample(&samples, sample_rate).unwrap()
    }
}

fn read_test_wav_raw() -> Vec<u8> {
    use phonex::audio::AudioPreprocessor;
    let (samples, sample_rate) = AudioPreprocessor::read_wav(&format!("{}/test_wavs/0.wav", MODEL_DIR)).unwrap();
    let samples = maybe_resample(samples, sample_rate, 16000);
    samples.into_iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[tokio::test]
#[ignore = "slow — loads ONNX model"]
async fn test_transcribe_real_audio() {
    let (engine, info) = real_engine();
    let app = phonex::server::app(engine, MODEL_DIR.to_string(), info);
    let audio_bytes = read_test_wav_raw();

    let boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"audio\"; filename=\"test.raw\"\r\n\r\n");
    body.extend_from_slice(&audio_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/transcribe")
                .method("POST")
                .header("content-type", format!("multipart/form-data; boundary={boundary}"))
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let text = json["text"].as_str().unwrap();
    assert!(!text.is_empty(), "transcription should not be empty");
    assert!(text.contains("เกม") || text.contains("อินโดนีเซีย"), "unexpected transcription: {}", text);
}

#[tokio::test]
#[ignore = "slow — loads ONNX model"]
async fn test_transcribe_stream_real_audio() {
    let (engine, info) = real_engine();
    let app = phonex::server::app(engine, MODEL_DIR.to_string(), info);
    let audio_bytes = read_test_wav_raw();

    let boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"audio\"; filename=\"test.raw\"\r\n\r\n");
    body.extend_from_slice(&audio_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/transcribe/stream")
                .method("POST")
                .header("content-type", format!("multipart/form-data; boundary={boundary}"))
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("data:"), "SSE should contain data events");
}
