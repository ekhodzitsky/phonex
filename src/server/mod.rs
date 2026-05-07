//! HTTP + WebSocket server that accepts audio and streams transcripts.
//!
//! Single port serves both REST API (health, transcribe, SSE) and WebSocket.

pub mod http;
pub mod metrics;
pub mod rate_limit;
pub mod ws;

#[cfg(feature = "grpc")]
pub mod grpc;

use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::Instrument;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::inference::Engine;

/// Supported input sample rates (Hz). Default is 16000.
pub(crate) use crate::inference::SUPPORTED_RATES;

/// Hint (milliseconds) returned to clients that hit pool backpressure.
pub(crate) const POOL_RETRY_AFTER_MS: u32 = 30_000;
pub(crate) const POOL_RETRY_AFTER_SECS: u64 = 30;

/// Runtime limits for the server.
#[derive(Debug, Clone)]
pub struct RuntimeLimits {
    pub body_limit_bytes: usize,
    pub rate_limit_per_minute: u32,
    pub rate_limit_burst: u32,
    pub max_ws_connections: usize,
    pub api_key: Option<String>,
    pub admin_api_key: Option<String>,
    pub trust_proxy: bool,
    pub cors_origins: Vec<String>,
    pub ws_idle_timeout_secs: u64,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            body_limit_bytes: 500 * 1024 * 1024, // 500 MB
            rate_limit_per_minute: 0,            // disabled by default
            rate_limit_burst: 10,
            max_ws_connections: 100,
            api_key: None,
            admin_api_key: None,
            trust_proxy: false,
            cors_origins: vec![
                "http://localhost:3000".into(),
                "http://localhost:5173".into(),
            ],
            ws_idle_timeout_secs: 60,
        }
    }
}

/// Async shutdown signal handler (SIGTERM / SIGINT).
pub async fn shutdown_signal() {
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("Failed to install SIGTERM handler");
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .expect("Failed to install SIGINT handler");

    tokio::select! {
        _ = sigterm.recv() => tracing::info!("Received SIGTERM"),
        _ = sigint.recv() => tracing::info!("Received SIGINT"),
    }
}

/// Build the axum router with all routes and middleware (default limits).
pub fn app(
    engine: Arc<Engine>,
    model_dir: String,
    model_info: crate::model_config::ModelInfo,
) -> Router {
    app_with_limits(
        engine,
        model_dir,
        model_info,
        RuntimeLimits::default(),
        tokio_util::sync::CancellationToken::new(),
    )
}

/// Build the axum router with all routes and middleware.
pub fn app_with_limits(
    engine: Arc<Engine>,
    model_dir: String,
    model_info: crate::model_config::ModelInfo,
    limits: RuntimeLimits,
    shutdown: tokio_util::sync::CancellationToken,
) -> Router {
    let metrics_registry = Arc::new(metrics::MetricsRegistry::new());
    metrics_registry.register_counter("requests_total", "Total HTTP requests");
    metrics_registry.register_histogram(
        "request_duration_seconds",
        "HTTP request duration",
        metrics::DEFAULT_BUCKETS,
    );
    metrics_registry.register_counter("ws_connections_total", "Total WebSocket connections");
    metrics_registry.register_counter("ws_messages_total", "Total WebSocket messages");

    let ws_semaphore = Arc::new(tokio::sync::Semaphore::new(limits.max_ws_connections));

    let tracker = tokio_util::task::TaskTracker::new();

    let state = Arc::new(http::AppState {
        engine: Arc::new(tokio::sync::RwLock::new(engine)),
        limits: limits.clone(),
        metrics_registry: Some(metrics_registry.clone()),
        shutdown,
        tracker,
        model_dir: Arc::new(tokio::sync::RwLock::new(model_dir)),
        model_info: Arc::new(tokio::sync::RwLock::new(model_info)),
        ws_semaphore,
    });

    let allow_origins: Vec<axum::http::HeaderValue> = limits
        .cors_origins
        .iter()
        .filter_map(|o| o.parse().ok())
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(allow_origins))
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
            axum::http::header::HeaderName::from_static("x-request-id"),
        ]);

    #[derive(OpenApi)]
    #[openapi(
        paths(
            http::health,
            http::models,
            http::transcribe,
        ),
        components(schemas(
            http::HealthResponse,
            http::ModelInfoResponse,
            http::TranscribeResponse,
            crate::inference::WordInfo,
        )),
        tags((name = "phonex", description = "Speech-to-text API"))
    )]
    struct ApiDoc;

    let mut router = Router::new()
        .route("/health", axum::routing::get(http::health))
        .route("/v1/models", axum::routing::get(http::models))
        .route("/v1/transcribe", axum::routing::post(http::transcribe))
        .route(
            "/v1/transcribe/batch",
            axum::routing::post(http::transcribe_batch),
        )
        .route(
            "/v1/transcribe/stream",
            axum::routing::post(http::transcribe_stream),
        )
        .route(
            "/v1/transcribe/stream",
            axum::routing::get(ws::ws_v1_transcribe_stream),
        )
        .route("/v1/admin/reload", axum::routing::post(http::reload))
        .route("/metrics", axum::routing::get(http::metrics))
        .route("/stream", axum::routing::get(ws::ws_handler))
        .merge(SwaggerUi::new("/docs").url("/openapi.json", ApiDoc::openapi()))
        .layer(DefaultBodyLimit::max(limits.body_limit_bytes))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    if limits.rate_limit_per_minute > 0 {
        let limiter = Arc::new(rate_limit::RateLimiter::new(
            limits.rate_limit_per_minute,
            limits.rate_limit_burst,
            limits.trust_proxy,
        ));
        router = router.layer(axum::middleware::from_fn_with_state(
            limiter,
            rate_limit::rate_limit_middleware,
        ));
    }

    // Request ID middleware (outermost).
    router = router.layer(axum::middleware::from_fn(request_id_middleware));

    // Auth middleware runs before rate limiting.
    router = router.layer(axum::middleware::from_fn_with_state(
        state.clone(),
        auth_middleware,
    ));

    router
}

/// Request ID middleware — attaches a unique request ID to every request.
/// The ID is propagated via `x-request-id` header (response) and tracing span.
pub async fn request_id_middleware(req: axum::extract::Request, next: Next) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span =
        tracing::info_span!("request", %request_id, method = %req.method(), uri = %req.uri());
    let mut response = next.run(req).instrument(span).await;
    response
        .headers_mut()
        .insert("x-request-id", request_id.parse().unwrap());
    response
}

/// API key authentication middleware.
/// If no API key is configured, all requests are allowed.
pub async fn auth_middleware(
    State(state): State<Arc<http::AppState>>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    let path = req.uri().path();
    let is_admin = path == "/v1/admin/reload" || path == "/metrics";
    let expected_key = if is_admin {
        state
            .limits
            .admin_api_key
            .as_ref()
            .or(state.limits.api_key.as_ref())
    } else {
        state.limits.api_key.as_ref()
    };

    if let Some(expected_key) = expected_key {
        let valid = req
            .headers()
            .get("authorization")
            .and_then(|h| h.to_str().ok())
            .is_some_and(|s| {
                s.strip_prefix("Bearer ")
                    .is_some_and(|token| token == expected_key)
            });
        if !valid {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::CONTENT_TYPE, "application/json")],
                Json(serde_json::json!({
                    "error": "Invalid API key",
                    "code": "invalid_api_key",
                })),
            )
                .into_response();
        }
    }
    next.run(req).await
}
