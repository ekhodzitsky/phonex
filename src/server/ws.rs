//! WebSocket handler for real-time streaming transcription.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use futures_util::StreamExt;
use std::sync::Arc;

use crate::inference::Engine;
use crate::protocol::{ClientMessage, ServerMessage};
use crate::server::http::AppState;

/// GET /v1/transcribe/stream — WebSocket upgrade for real-time streaming transcription.
pub async fn ws_v1_transcribe_stream(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_v1_stream(socket, state))
}

async fn handle_v1_stream(mut socket: WebSocket, state: Arc<AppState>) {
    // Enforce max concurrent WebSocket connections.
    let _permit = if state.limits.max_ws_connections > 0 {
        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            state.ws_semaphore.acquire(),
        )
        .await
        {
            Ok(Ok(p)) => Some(p),
            _ => {
                let _ = send_error(
                    &mut socket,
                    "Too many concurrent streaming connections",
                    "max_connections",
                    None,
                )
                .await;
                return;
            }
        }
    } else {
        None
    };

    // Build a StreamingPipeline for this connection.
    let mut pipeline = match crate::streaming_pipeline::StreamingPipeline::from_model_dir(
        &*state.model_dir.read().await,
        &*state.model_info.read().await,
        None,
    ) {
        Ok(p) => p,
        Err(e) => {
            let _ = send_error(
                &mut socket,
                &format!("Failed to load streaming pipeline: {e}"),
                "pipeline_error",
                None,
            )
            .await;
            return;
        }
    };

    if send_ready_message(&mut socket, &state).await.is_err() {
        return;
    }

    if let Some(ref registry) = state.metrics_registry {
        registry.counter_inc("ws_connections_total", vec![], 1);
    }

    let mut flushed = false;
    let mut pending_samples: usize = 0;
    const MAX_PENDING_SAMPLES: usize = 480_000; // 30 seconds at 16kHz

    loop {
        if state.shutdown.is_cancelled() {
            break;
        }

        let msg = match tokio::time::timeout(
            std::time::Duration::from_secs(state.limits.ws_idle_timeout_secs),
            socket.next(),
        ).await {
            Ok(Some(Ok(msg))) => msg,
            Ok(Some(Err(_)) | None) => break,
            Err(_) => break,
        };

        match msg {
            Message::Binary(data) => {
                if let Some(ref registry) = state.metrics_registry {
                    registry.counter_inc("ws_messages_total", vec![], 1);
                }

                let samples = match bytes_to_f32_samples(&data) {
                    Ok(s) => s,
                    Err(code) => {
                        let _ = send_error(&mut socket, "Invalid audio samples", code, None).await;
                        continue;
                    }
                };

                pending_samples += samples.len();
                if pending_samples > MAX_PENDING_SAMPLES {
                    let _ = send_error(
                        &mut socket,
                        "Audio buffer full",
                        "backpressure",
                        Some(5000),
                    ).await;
                    break;
                }

                match pipeline.accept_audio(&samples) {
                    Ok(new_tokens) => {
                        if !new_tokens.is_empty() {
                            let text = pipeline.text();
                            let timestamp = crate::inference::streaming::now_timestamp();
                            let reply = ServerMessage::Partial {
                                text,
                                timestamp,
                                words: vec![],
                            };
                            if socket
                                .send(Message::Text(
                                    serde_json::to_string(&reply).unwrap().into(),
                                ))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = send_error(
                            &mut socket,
                            &format!("Inference error: {e}"),
                            "inference_error",
                            None,
                        )
                        .await;
                    }
                }
            }
            Message::Text(text) => {
                if let Some(ref registry) = state.metrics_registry {
                    registry.counter_inc("ws_messages_total", vec![], 1);
                }

                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::Clear) => {
                        pipeline.reset();
                        pending_samples = 0;
                    }
                    Ok(ClientMessage::Stop) => {
                        flush_and_send_final(&mut socket, &mut pipeline).await;
                        flushed = true;
                        break;
                    }
                    Ok(ClientMessage::Configure { sample_rate }) => {
                        if let Some(rate) = sample_rate
                            && !crate::server::SUPPORTED_RATES.contains(&rate) {
                                let _ = send_error(
                                    &mut socket,
                                    &format!("Unsupported sample rate: {rate}"),
                                    "invalid_sample_rate",
                                    None,
                                )
                                .await;
                            }
                    }
                    Err(_) => {
                        // Plain-text commands for simple clients.
                        match text.trim() {
                            "CLEAR" => {
                                pipeline.reset();
                                pending_samples = 0;
                            }
                            "STOP" => {
                                flush_and_send_final(&mut socket, &mut pipeline).await;
                                flushed = true;
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    // Connection dropped without explicit STOP — flush remaining audio.
    if !flushed {
        flush_and_send_final(&mut socket, &mut pipeline).await;
    }
}

async fn flush_and_send_final(socket: &mut WebSocket, pipeline: &mut crate::streaming_pipeline::StreamingPipeline) {
    match pipeline.flush() {
        Ok(text) => {
            let timestamp = crate::inference::streaming::now_timestamp();
            let msg = ServerMessage::Final {
                text,
                timestamp,
                words: vec![],
            };
            let _ = socket
                .send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
                .await;
        }
        Err(e) => {
            let _ = send_error(
                socket,
                &format!("Flush error: {e}"),
                "flush_error",
                None,
            )
            .await;
        }
    }
}

fn bytes_to_f32_samples(data: &[u8]) -> Result<Vec<f32>, &'static str> {
    let samples = crate::inference::audio::bytes_to_f32_samples(data);
    if samples.iter().any(|s| !s.is_finite()) {
        return Err("invalid_audio_samples");
    }
    Ok(samples)
}

// Keep the existing /stream handler below ...

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    if send_ready_message(&mut socket, &state).await.is_err() {
        return;
    }

    if let Some(ref registry) = state.metrics_registry {
        registry.counter_inc("ws_connections_total", vec![], 1);
    }

    let engine = state.engine.read().await.clone();

    let mut stream_state = match engine.create_state(false) {
        Ok(s) => s,
        Err(e) => {
            let _ = send_error(&mut socket, &format!("{e}"), "state_error", None).await;
            return;
        }
    };

    // Checkout a triplet from the pool for this connection
    let (mut triplet, reservation) = match checkout_triplet(&engine).await {
        Ok(t) => t,
        Err(_) => {
            let _ = send_error(&mut socket, "Server busy", "pool_timeout", Some(30_000)).await;
            return;
        }
    };

    let mut flushed = false;

    loop {
        if state.shutdown.is_cancelled() {
            break;
        }

        let msg = match tokio::time::timeout(
            std::time::Duration::from_secs(state.limits.ws_idle_timeout_secs),
            socket.next(),
        ).await {
            Ok(Some(Ok(msg))) => msg,
            Ok(Some(Err(_)) | None) => break,
            Err(_) => break,
        };

        match msg {
            Message::Binary(data) => {
                if let Some(ref registry) = state.metrics_registry {
                    registry.counter_inc("ws_messages_total", vec![], 1);
                }

                let samples = match bytes_to_f32_samples(&data) {
                    Ok(s) => s,
                    Err(code) => {
                        let _ = send_error(&mut socket, "Invalid audio samples", code, None).await;
                        continue;
                    }
                };

                // Push to streaming state
                stream_state.audio_buffer.extend_from_slice(&samples);

                // Check if we should process
                if stream_state.should_process() {
                    match engine.process_chunk(&[], &mut stream_state, &mut triplet) {
                        Ok(segments) => {
                            for seg in segments {
                                let msg = if seg.is_final {
                                    ServerMessage::Final {
                                        text: seg.text.to_string(),
                                        timestamp: seg.timestamp,
                                        words: seg.words.to_vec(),
                                    }
                                } else {
                                    ServerMessage::Partial {
                                        text: seg.text.to_string(),
                                        timestamp: seg.timestamp,
                                        words: seg.words.to_vec(),
                                    }
                                };
                                if socket
                                    .send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = send_error(&mut socket, &format!("{e}"), "inference_error", None).await;
                        }
                    }
                }
            }
            Message::Text(text) => {
                if let Some(ref registry) = state.metrics_registry {
                    registry.counter_inc("ws_messages_total", vec![], 1);
                }

                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::Clear) => {
                        stream_state.clear();
                    }
                    Ok(ClientMessage::Stop) => {
                        if let Some(seg) = engine.flush_state(&mut stream_state, &mut triplet) {
                            let msg = ServerMessage::Final {
                                text: seg.text.to_string(),
                                timestamp: seg.timestamp,
                                words: seg.words.to_vec(),
                            };
                            let _ = socket
                                .send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
                                .await;
                        }
                        flushed = true;
                        break;
                    }
                    Ok(ClientMessage::Configure { sample_rate }) => {
                        if let Some(rate) = sample_rate
                            && !crate::server::SUPPORTED_RATES.contains(&rate) {
                                let _ = send_error(
                                    &mut socket,
                                    &format!("Unsupported sample rate: {rate}"),
                                    "invalid_sample_rate",
                                    None,
                                )
                                .await;
                            }
                    }
                    Err(_) => {
                        // Plain text commands for simple clients
                        match text.trim() {
                            "CLEAR" => {
                                stream_state.clear();
                            }
                            "STOP" => {
                                if let Some(seg) =
                                    engine.flush_state(&mut stream_state, &mut triplet)
                                {
                                    let msg = ServerMessage::Final {
                                        text: seg.text.to_string(),
                                        timestamp: seg.timestamp,
                                        words: seg.words.to_vec(),
                                    };
                                    let _ = socket
                                        .send(Message::Text(
                                            serde_json::to_string(&msg).unwrap().into(),
                                        ))
                                        .await;
                                }
                                flushed = true;
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    if !flushed
        && let Some(seg) = engine.flush_state(&mut stream_state, &mut triplet)
    {
        let msg = ServerMessage::Final {
            text: seg.text.to_string(),
            timestamp: seg.timestamp,
            words: seg.words.to_vec(),
        };
        let _ = socket
            .send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
            .await;
    }

    // Return triplet to pool
    reservation.checkin(triplet);
}

async fn checkout_triplet(
    engine: &Arc<Engine>,
) -> Result<
    (
        crate::inference::SessionTriplet,
        crate::inference::OwnedReservation<crate::inference::SessionTriplet>,
    ),
    (), // FIXME: proper error type
> {
    match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        engine.pool.checkout(),
    )
    .await
    {
        Ok(Ok(guard)) => {
            let (triplet, reservation) = guard.into_owned();
            Ok((triplet, reservation))
        }
        Ok(Err(_)) => Err(()),
        Err(_) => Err(()),
    }
}

async fn send_ready_message(
    socket: &mut WebSocket,
    state: &Arc<AppState>,
) -> Result<(), axum::Error> {
    let ready = ServerMessage::Ready {
        model: state.model_info.read().await.model_id.clone(),
        sample_rate: crate::inference::TARGET_SAMPLE_RATE,
        version: env!("CARGO_PKG_VERSION").into(),
        supported_rates: crate::server::SUPPORTED_RATES.to_vec(),
    };
    socket
        .send(Message::Text(serde_json::to_string(&ready).unwrap().into()))
        .await
}

async fn send_error(
    socket: &mut WebSocket,
    message: &str,
    code: &str,
    retry_after_ms: Option<u32>,
) -> Result<(), axum::Error> {
    let msg = ServerMessage::Error {
        message: message.into(),
        code: code.into(),
        retry_after_ms,
    };
    socket
        .send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
        .await
}
