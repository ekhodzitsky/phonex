//! Integration tests for the HTTP server endpoints.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use phonex::inference::Engine;
use phonex::inference::pool::SessionPool;
use phonex::tokenizer::Tokenizer;
use std::sync::Arc;

const MODEL_DIR: &str = "models/sherpa-onnx-zipformer-thai-2024-06-20";

macro_rules! skip_if_no_models {
    () => {
        if !std::path::Path::new(MODEL_DIR).is_dir() {
            eprintln!("Skipping test: model directory not found");
            return;
        }
    };
}

fn test_engine() -> (Arc<Engine>, phonex::model_config::ModelInfo) {
    let paths = phonex::model_config::discover_model_files(MODEL_DIR).unwrap();
    let tokenizer = Arc::new(
        Tokenizer::from_file(
            paths.tokenizer.to_str().unwrap_or(""),
            paths.tokens.to_str().unwrap_or(""),
            0,
        )
        .unwrap(),
    );
    let pool = SessionPool::new(vec![]);
    let info = phonex::model_config::ModelInfo {
        sample_rate: 16000,
        n_mels: 80,
        blank_id: 0,
        context_size: 2,
        d_model: 512,
        vocab_size: 2000,
        encoder_inputs: vec!["x".into()],
        encoder_outputs: vec!["encoder_out".into()],
        decoder_inputs: vec!["decoder_input".into()],
        decoder_outputs: vec!["decoder_out".into()],
        joiner_inputs: vec!["encoder_out".into(), "decoder_out".into()],
        joiner_outputs: vec!["logits".into()],
        model_id: "sherpa-onnx-zipformer-th".into(),
        model_name: "sherpa-onnx-zipformer-thai-2024-06-20".into(),
    };
    (Arc::new(Engine::new(pool, tokenizer, info.clone())), info)
}

#[tokio::test]
async fn test_health() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let app = phonex::server::app(engine, MODEL_DIR.to_string(), info);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
    assert_eq!(json["model"], "sherpa-onnx-zipformer-th");
}

#[tokio::test]
async fn test_models() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let app = phonex::server::app(engine, MODEL_DIR.to_string(), info);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["id"], "sherpa-onnx-zipformer-th");
    assert_eq!(json["vocab_size"], 2000);
    assert_eq!(json["sample_rate"], 16000);
}

#[tokio::test]
async fn test_metrics() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let app = phonex::server::app(engine, MODEL_DIR.to_string(), info);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("# HELP"));
    assert!(text.contains("# TYPE"));
}

#[tokio::test]
async fn test_transcribe_empty_body() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let app = phonex::server::app(engine, MODEL_DIR.to_string(), info);

    let boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"test.raw\"\r\n\r\n\r\n--{boundary}--\r\n"
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/transcribe")
                .method("POST")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    // Empty body should return 400 bad request (or 422)
    assert!(response.status().is_client_error());
}

#[tokio::test]
async fn test_reload_bad_model_dir() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let app = phonex::server::app(engine, MODEL_DIR.to_string(), info);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/admin/reload?model_dir=/nonexistent")
                .method("POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_transcribe_unauthorized() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let limits = phonex::server::RuntimeLimits {
        api_key: Some("secret".into()),
        ..phonex::server::RuntimeLimits::default()
    };
    let app = phonex::server::app_with_limits(
        engine,
        MODEL_DIR.to_string(),
        info,
        limits,
        tokio_util::sync::CancellationToken::new(),
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/transcribe")
                .method("POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_transcribe_authorized() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let limits = phonex::server::RuntimeLimits {
        api_key: Some("secret".into()),
        ..phonex::server::RuntimeLimits::default()
    };
    let app = phonex::server::app_with_limits(
        engine,
        MODEL_DIR.to_string(),
        info,
        limits,
        tokio_util::sync::CancellationToken::new(),
    );

    let boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"test.raw\"\r\n\r\n\r\n--{boundary}--\r\n"
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/transcribe")
                .method("POST")
                .header("authorization", "Bearer secret")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should not be 401; empty body yields 400/422.
    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_admin_reload_unauthorized() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let limits = phonex::server::RuntimeLimits {
        admin_api_key: Some("admin_secret".into()),
        ..phonex::server::RuntimeLimits::default()
    };
    let app = phonex::server::app_with_limits(
        engine,
        MODEL_DIR.to_string(),
        info,
        limits,
        tokio_util::sync::CancellationToken::new(),
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/admin/reload")
                .method("POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_cors_preflight() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let app = phonex::server::app(engine, MODEL_DIR.to_string(), info);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/transcribe")
                .method("OPTIONS")
                .header("origin", "http://localhost:3000")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(
        response.status() == StatusCode::OK || response.status() == StatusCode::NO_CONTENT,
        "expected 200 or 204, got {:?}",
        response.status()
    );
    assert!(
        response
            .headers()
            .contains_key("access-control-allow-origin"),
        "missing Access-Control-Allow-Origin header"
    );
}
