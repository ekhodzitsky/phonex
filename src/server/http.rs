//! HTTP handlers for REST API endpoints.

use axum::body::Bytes;
use axum::extract::{Multipart, Query, State};
use axum::http::StatusCode;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::metrics::MetricsRegistry;
use super::{RuntimeLimits, POOL_RETRY_AFTER_MS, POOL_RETRY_AFTER_SECS};
use crate::inference::Engine;

/// Shared application state for all handlers.
pub struct AppState {
    pub engine: Arc<tokio::sync::RwLock<Arc<Engine>>>,
    pub limits: RuntimeLimits,
    pub metrics_registry: Option<Arc<MetricsRegistry>>,
    pub shutdown: tokio_util::sync::CancellationToken,
    pub tracker: tokio_util::task::TaskTracker,
    pub model_dir: Arc<tokio::sync::RwLock<String>>,
    pub model_info: Arc<tokio::sync::RwLock<crate::model_config::ModelInfo>>,
    pub ws_semaphore: Arc<tokio::sync::Semaphore>,
}

/// GET /metrics — Prometheus text-format exposition.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    match &state.metrics_registry {
        Some(registry) => (
            StatusCode::OK,
            [(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            registry.render_prometheus(),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "metrics endpoint disabled",
                "code": "metrics_disabled",
            })),
        )
            .into_response(),
    }
}

/// Health check response.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub version: String,
}

/// Model info response.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ModelInfoResponse {
    pub id: String,
    pub name: String,
    pub version: String,
    pub encoder: String,
    pub vocab_size: usize,
    pub sample_rate: u32,
    pub pool_size: usize,
    pub pool_available: usize,
    pub supported_formats: Vec<String>,
    pub supported_rates: Vec<u32>,
}

/// Transcription response.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct TranscribeResponse {
    pub text: String,
    pub words: Vec<crate::inference::WordInfo>,
    pub duration: f64,
}

type ApiError = Response;

fn api_error(status: StatusCode, msg: &str, code: &str) -> ApiError {
    (status, Json(serde_json::json!({"error": msg, "code": code}))).into_response()
}

fn api_timeout_error() -> ApiError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, POOL_RETRY_AFTER_SECS.to_string())],
        Json(serde_json::json!({
            "error": "Server busy, try again later",
            "code": "timeout",
            "retry_after_ms": POOL_RETRY_AFTER_MS,
        })),
    )
        .into_response()
}

fn api_pool_closed_error() -> ApiError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": "Server is shutting down",
            "code": "pool_closed",
        })),
    )
        .into_response()
}

async fn checkout_triplet(
    engine: &Arc<Engine>,
) -> Result<
    (
        crate::inference::SessionTriplet,
        crate::inference::OwnedReservation<crate::inference::SessionTriplet>,
    ),
    ApiError,
> {
    match tokio::time::timeout(std::time::Duration::from_secs(30), engine.pool.checkout()).await {
        Ok(Ok(guard)) => {
            let (triplet, reservation) = guard.into_owned();
            Ok((triplet, reservation))
        }
        Ok(Err(_pool_closed)) => Err(api_pool_closed_error()),
        Err(_timeout) => Err(api_timeout_error()),
    }
}

/// Guard that records HTTP request metrics on drop.
struct MetricsGuard<'a> {
    registry: &'a Option<Arc<MetricsRegistry>>,
    method: &'static str,
    path: &'static str,
    start: std::time::Instant,
}

impl Drop for MetricsGuard<'_> {
    fn drop(&mut self) {
        if let Some(r) = self.registry {
            let labels = vec![
                ("method".to_string(), self.method.to_string()),
                ("path".to_string(), self.path.to_string()),
            ];
            r.counter_inc("requests_total", labels.clone(), 1);
            r.histogram_record("request_duration_seconds", labels, self.start.elapsed().as_secs_f64());
        }
    }
}

/// GET /health — health check for monitoring and Docker HEALTHCHECK.
#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse),
    )
)]
#[tracing::instrument(skip(state))]
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let _guard = MetricsGuard {
        registry: &state.metrics_registry,
        method: "GET",
        path: "/health",
        start: std::time::Instant::now(),
    };
    let engine = state.engine.read().await;
    let model_info = state.model_info.read().await;
    let status = if engine.pool.available() > 0 || engine.pool.total() == 0 {
        "ok"
    } else {
        "degraded"
    };
    Json(HealthResponse {
        status: status.into(),
        model: model_info.model_id.clone(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

/// GET /v1/models — list loaded models and capabilities.
#[utoipa::path(
    get,
    path = "/v1/models",
    responses(
        (status = 200, description = "Model information", body = ModelInfoResponse),
    )
)]
#[tracing::instrument(skip(state))]
pub async fn models(State(state): State<Arc<AppState>>) -> Json<ModelInfoResponse> {
    let _guard = MetricsGuard {
        registry: &state.metrics_registry,
        method: "GET",
        path: "/v1/models",
        start: std::time::Instant::now(),
    };
    let engine = state.engine.read().await;
    let model_info = state.model_info.read().await;
    Json(ModelInfoResponse {
        id: model_info.model_id.clone(),
        name: model_info.model_name.clone(),
        version: env!("CARGO_PKG_VERSION").into(),
        encoder: "int8".into(),
        vocab_size: engine.vocab_size(),
        sample_rate: crate::inference::TARGET_SAMPLE_RATE,
        pool_size: engine.pool.total(),
        pool_available: engine.pool.available(),
        supported_formats: vec![
            "raw-f32le".into(),
        ],
        supported_rates: super::SUPPORTED_RATES.to_vec(),
    })
}

#[derive(Debug, Deserialize)]
pub struct TranscribeQuery {
    #[serde(default)]
    pub sample_rate: Option<u32>,
    #[serde(default)]
    pub vad: bool,
}

async fn extract_audio_from_multipart(
    multipart: &mut Multipart,
    query: &TranscribeQuery,
    body_limit_bytes: usize,
) -> Result<(Bytes, u32), ApiError> {
    let mut audio_bytes: Option<Bytes> = None;
    let mut sample_rate = query.sample_rate;

    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("audio") => {
                if let Ok(data) = field.bytes().await {
                    audio_bytes = Some(data);
                }
            }
            Some("sample_rate") => {
                if let Ok(text) = field.text().await
                    && let Ok(rate) = text.trim().parse::<u32>() {
                        sample_rate = Some(rate);
                    }
            }
            _ => {}
        }
    }

    let body = audio_bytes.ok_or_else(|| api_error(
        StatusCode::BAD_REQUEST,
        "Missing 'audio' field in multipart upload",
        "missing_audio",
    ))?;

    if body.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
    }

    if body.len() > body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }

    let client_rate = sample_rate.unwrap_or(crate::inference::TARGET_SAMPLE_RATE);
    if !super::SUPPORTED_RATES.contains(&client_rate) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            &format!("Unsupported sample rate: {client_rate}Hz"),
            "invalid_sample_rate",
        ));
    }

    Ok((body, client_rate))
}

/// POST /v1/transcribe — upload audio via multipart, get full transcript.
///
/// Accepts `audio` field with raw mono f32 LE bytes and optional `sample_rate` field.
/// Default sample rate is 16000 Hz.
#[utoipa::path(
    post,
    path = "/v1/transcribe",
    responses(
        (status = 200, description = "Transcription result", body = TranscribeResponse),
        (status = 503, description = "Server busy"),
    )
)]
#[tracing::instrument(skip(state, multipart))]
pub async fn transcribe(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TranscribeQuery>,
    mut multipart: Multipart,
) -> Result<Json<TranscribeResponse>, ApiError> {
    let _guard = MetricsGuard {
        registry: &state.metrics_registry,
        method: "POST",
        path: "/v1/transcribe",
        start: std::time::Instant::now(),
    };
    let (body, client_rate) = extract_audio_from_multipart(&mut multipart, &query, state.limits.body_limit_bytes).await?;
    let use_vad = query.vad;
    let engine = state.engine.read().await.clone();
    let (triplet, reservation) = checkout_triplet(&engine).await?;

    let result = tokio::task::spawn_blocking(move || {
        let mut triplet = triplet;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Convert f32 LE bytes to Vec<f32>
            let samples_f32 = crate::inference::audio::bytes_to_f32_samples(&body);
            // Resample if needed
            let samples = if client_rate == crate::inference::TARGET_SAMPLE_RATE {
                samples_f32
            } else {
                crate::inference::audio::resample(&samples_f32, client_rate, crate::inference::TARGET_SAMPLE_RATE)
                    .unwrap_or_default()
            };
            if use_vad {
                engine.transcribe_samples_with_vad(&samples, &mut triplet)
            } else {
                engine.transcribe_samples(&samples, &mut triplet)
            }
        }));
        match r {
            Ok(inference_result) => (inference_result, triplet),
            Err(_) => {
                tracing::error!("Panic in REST transcribe — triplet recovered");
                (
                    Err(crate::SiamError::Inference("Inference thread panicked".into())),
                    triplet,
                )
            }
        }
    })
    .await;

    match result {
        Ok((Ok(result), triplet)) => {
            reservation.checkin(triplet);
            Ok(Json(TranscribeResponse {
                text: result.text,
                words: result.words,
                duration: result.duration_s,
            }))
        }
        Ok((Err(e), triplet)) => {
            reservation.checkin(triplet);
            tracing::error!("Transcription error: {e}");
            Err(api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Transcription failed. Check audio format.",
                "transcription_error",
            ))
        }
        Err(e) => {
            tracing::error!("spawn_blocking join error: {e}");
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error",
                "internal",
            ))
        }
    }
}

/// POST /v1/transcribe/batch — upload multiple audio files, get transcripts for all.
#[tracing::instrument(skip(state, multipart))]
pub async fn transcribe_batch(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TranscribeQuery>,
    mut multipart: Multipart,
) -> Result<Json<Vec<TranscribeResponse>>, ApiError> {
    let _guard = MetricsGuard {
        registry: &state.metrics_registry,
        method: "POST",
        path: "/v1/transcribe/batch",
        start: std::time::Instant::now(),
    };
    let mut files: Vec<(Bytes, u32)> = Vec::new();
    let default_rate = query.sample_rate.unwrap_or(crate::inference::TARGET_SAMPLE_RATE);

    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("audio")
            && let Ok(data) = field.bytes().await {
                files.push((data, default_rate));
            }
    }

    if files.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Missing 'audio' field(s) in multipart upload",
            "missing_audio",
        ));
    }

    let engine = state.engine.read().await.clone();
    let (triplet, reservation) = checkout_triplet(&engine).await?;

    let results = tokio::task::spawn_blocking(move || {
        let mut triplet = triplet;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Convert all files to f32 samples
            let mut sample_buffers: Vec<Vec<f32>> = Vec::with_capacity(files.len());
            for (body, client_rate) in &files {
                let samples_f32 = crate::inference::audio::bytes_to_f32_samples(body);
                let samples = if *client_rate == crate::inference::TARGET_SAMPLE_RATE {
                    samples_f32
                } else {
                    crate::inference::audio::resample(&samples_f32, *client_rate, crate::inference::TARGET_SAMPLE_RATE)
                        .unwrap_or_default()
                };
                sample_buffers.push(samples);
            }

            let refs: Vec<&[f32]> = sample_buffers.iter().map(|s| s.as_slice()).collect();
            engine.transcribe_batch(refs, &mut triplet)
        }));

        match r {
            Ok(Ok(batch_results)) => {
                reservation.checkin(triplet);
                batch_results.into_iter().map(|result| TranscribeResponse {
                    text: result.text,
                    words: result.words,
                    duration: result.duration_s,
                }).collect::<Vec<_>>()
            }
            Ok(Err(e)) => {
                reservation.checkin(triplet);
                tracing::error!("Batch transcription error: {e}");
                vec![TranscribeResponse {
                    text: String::new(),
                    words: vec![],
                    duration: 0.0,
                }]
            }
            Err(_) => {
                reservation.checkin(triplet);
                tracing::error!("Panic in batch transcription");
                vec![TranscribeResponse {
                    text: String::new(),
                    words: vec![],
                    duration: 0.0,
                }]
            }
        }
    })
    .await
    .map_err(|e| {
        tracing::error!("spawn_blocking join error: {e}");
        api_error(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error", "internal")
    })?;

    Ok(Json(results))
}

/// POST /v1/transcribe/stream — upload audio via multipart, get SSE stream of partial/final results.
#[tracing::instrument(skip(state, multipart))]
pub async fn transcribe_stream(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TranscribeQuery>,
    mut multipart: Multipart,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    let _guard = MetricsGuard {
        registry: &state.metrics_registry,
        method: "POST",
        path: "/v1/transcribe/stream",
        start: std::time::Instant::now(),
    };
    let (body, client_rate) = extract_audio_from_multipart(&mut multipart, &query, state.limits.body_limit_bytes).await?;
    let engine = state.engine.read().await.clone();
    let (triplet, reservation) = checkout_triplet(&engine).await?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<crate::inference::TranscriptSegment, String>>(16);

    let engine = engine.clone();
    let cancel = state.shutdown.clone();
    let tracker = state.tracker.clone();
    tracker.spawn_blocking(move || {
        let mut triplet = triplet;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let samples_f32 = crate::inference::audio::bytes_to_f32_samples(&body);
            let samples = if client_rate == crate::inference::TARGET_SAMPLE_RATE {
                samples_f32
            } else {
                crate::inference::audio::resample(&samples_f32, client_rate, crate::inference::TARGET_SAMPLE_RATE)
                    .unwrap_or_default()
            };

            let mut stream_state = match engine.create_state(false) {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.blocking_send(Err(format!("{e}")));
                    return;
                }
            };

            let chunk_size = crate::inference::TARGET_SAMPLE_RATE as usize;
            for chunk in samples.chunks(chunk_size) {
                if cancel.is_cancelled() {
                    return;
                }
                match engine.process_chunk(chunk, &mut stream_state, &mut triplet) {
                    Ok(segs) => {
                        for seg in segs {
                            if tx.blocking_send(Ok(seg)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("{e}")));
                        return;
                    }
                }
            }

            if !cancel.is_cancelled()
                && let Some(seg) = engine.flush_state(&mut stream_state, &mut triplet) {
                    let _ = tx.blocking_send(Ok(seg));
                }
        }));

        if result.is_err() {
            tracing::error!("Panic in SSE inference task — triplet recovered");
        }
        reservation.checkin(triplet);
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(|result| {
        let event = match result {
            Ok(seg) => {
                let msg = if seg.is_final {
                    serde_json::json!({"type": "final", "text": seg.text.as_ref(), "timestamp": seg.timestamp, "words": seg.words.as_ref()})
                } else {
                    serde_json::json!({"type": "partial", "text": seg.text.as_ref(), "timestamp": seg.timestamp, "words": seg.words.as_ref()})
                };
                Event::default().data(msg.to_string())
            }
            Err(_) => {
                let msg = serde_json::json!({"type": "error", "message": "Transcription failed.", "code": "inference_error"});
                Event::default().data(msg.to_string())
            }
        };
        Ok(event)
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}



/// POST /v1/admin/reload — hot-swap the loaded model without restarting the server.
#[derive(Debug, Deserialize)]
pub struct ReloadQuery {
    /// Optional new model directory. If omitted, reloads the current model.
    pub model_dir: Option<String>,
}

#[tracing::instrument(skip(state))]
pub async fn reload(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ReloadQuery>,
) -> Result<Json<HealthResponse>, ApiError> {
    let new_model_dir = query.model_dir.unwrap_or_else(|| {
        state.model_dir.blocking_read().clone()
    });

    tracing::info!(model_dir = %new_model_dir, "Hot-swapping model");

    let info = crate::model_config::ModelInfo::from_model_dir(&new_model_dir)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, &format!("Failed to load model: {e}"), "model_load_error"))?;
    let paths = crate::model_config::discover_model_files(&new_model_dir)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, &format!("Failed to discover model files: {e}"), "model_discovery_error"))?;

    let tokenizer = Arc::new(crate::tokenizer::Tokenizer::from_file(
        paths.tokenizer.to_str().unwrap_or(""),
        paths.tokens.to_str().unwrap_or(""),
        info.blank_id,
    ).map_err(|e| api_error(StatusCode::BAD_REQUEST, &format!("Failed to load tokenizer: {e}"), "tokenizer_error"))?);

    let pool_size = state.engine.read().await.pool.total().max(1);
    let mut triplets = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        tracing::debug!(slot = i, "Loading ONNX sessions for hot-swap");
        triplets.push(crate::inference::SessionTriplet::from_model_dir(&new_model_dir, &info)
            .map_err(|e| api_error(StatusCode::BAD_REQUEST, &format!("Failed to create session: {e}"), "session_error"))?);
    }

    let pool = crate::inference::pool::SessionPool::new(triplets);
    let new_engine = Arc::new(
        crate::inference::Engine::new(pool, tokenizer, info.clone()).with_vad(
            paths.vad.map(|p| p.to_string_lossy().to_string()).unwrap_or_default().as_str(),
        ),
    );

    {
        let mut engine_guard = state.engine.write().await;
        *engine_guard = new_engine;
    }
    {
        let mut model_dir_guard = state.model_dir.write().await;
        *model_dir_guard = new_model_dir;
    }
    {
        let mut model_info_guard = state.model_info.write().await;
        *model_info_guard = info;
    }

    let engine = state.engine.read().await;
    let model_info = state.model_info.read().await;
    let status = if engine.pool.available() > 0 || engine.pool.total() == 0 {
        "ok"
    } else {
        "degraded"
    };

    Ok(Json(HealthResponse {
        status: status.into(),
        model: model_info.model_id.clone(),
        version: env!("CARGO_PKG_VERSION").into(),
    }))
}
