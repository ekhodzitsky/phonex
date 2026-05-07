//! Integration tests for WebSocket streaming.

use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use phonex::inference::Engine;
use phonex::inference::pool::SessionPool;
use phonex::tokenizer::Tokenizer;

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
#[ignore = "requires ONNX model files"]
async fn test_ws_v1_stream_config_chunk_finalize() {
    skip_if_no_models!();
    let (engine, info) = test_engine();
    let app = phonex::server::app(engine, MODEL_DIR.to_string(), info);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give the server a moment to start
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let url = format!("ws://{}/v1/transcribe/stream", addr);
    let (mut ws_stream, _) = connect_async(&url).await.expect("Failed to connect");

    // Send config
    let config = serde_json::json!({
        "type": "config",
        "sample_rate": 16000,
        "language": "en",
    });
    ws_stream
        .send(Message::Text(config.to_string().into()))
        .await
        .unwrap();

    // Send a tiny chunk of silence (f32 LE)
    let silence: Vec<f32> = vec![0.0f32; 160]; // 10ms @ 16kHz
    let bytes: Vec<u8> = silence.iter().flat_map(|f| f.to_le_bytes()).collect();
    ws_stream.send(Message::Binary(bytes.into())).await.unwrap();

    // Send finalize
    let finalize = serde_json::json!({ "type": "finalize" });
    ws_stream
        .send(Message::Text(finalize.to_string().into()))
        .await
        .unwrap();

    // Collect responses
    let mut got_ready = false;
    let mut got_final = false;
    let timeout = tokio::time::Duration::from_secs(5);

    loop {
        let msg = tokio::time::timeout(timeout, ws_stream.next()).await;
        match msg {
            Ok(Some(Ok(Message::Text(text)))) => {
                let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
                if parsed["type"] == "ready" {
                    got_ready = true;
                } else if parsed["type"] == "final" {
                    got_final = true;
                    break;
                }
            }
            Ok(Some(Ok(Message::Close(_)))) | Ok(None) => break,
            Ok(Some(Ok(_))) => continue, // ignore ping/pong/binary
            Ok(Some(Err(e))) => panic!("WebSocket error: {e}"),
            Err(_) => panic!("Timeout waiting for WebSocket response"),
        }
    }

    assert!(got_ready, "Expected ready message");
    assert!(got_final, "Expected final message");
}
