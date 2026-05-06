//! Per-IP token-bucket rate limiter.

use axum::extract::{ConnectInfo, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};
use dashmap::DashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    refill_per_ms: f64,
    tokens: f64,
    last_refill: Instant,
    last_seen_ms: u64,
}

impl TokenBucket {
    pub fn new(capacity: u32, refill_per_ms: f64, now: Instant, now_ms: u64) -> Self {
        Self {
            capacity: capacity as f64,
            refill_per_ms,
            tokens: capacity as f64,
            last_refill: now,
            last_seen_ms: now_ms,
        }
    }

    pub fn try_consume(&mut self, now: Instant, now_ms: u64) -> bool {
        let elapsed_ms = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64()
            * 1000.0;
        if elapsed_ms > 0.0 {
            self.tokens = (self.tokens + elapsed_ms * self.refill_per_ms).min(self.capacity);
            self.last_refill = now;
        }
        self.last_seen_ms = now_ms;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

pub const MAX_RPM: u32 = 60_000;

pub struct RateLimiter {
    buckets: DashMap<IpAddr, TokenBucket>,
    capacity: u32,
    refill_per_ms: f64,
    effective_rpm: u32,
}

impl RateLimiter {
    pub fn new(rpm: u32, burst: u32) -> Self {
        if rpm > MAX_RPM {
            tracing::warn!(
                rpm,
                max_rpm = MAX_RPM,
                "rate_limit_per_minute exceeds {MAX_RPM}; clamped to {MAX_RPM}"
            );
        }
        let effective_rpm = rpm.min(MAX_RPM);
        let refill_per_ms = effective_rpm as f64 / 60_000.0;
        Self {
            buckets: DashMap::new(),
            capacity: burst.max(1),
            refill_per_ms,
            effective_rpm,
        }
    }

    pub fn interval_ms(&self) -> u64 {
        (60_000u64 / self.effective_rpm.max(1) as u64).max(1)
    }

    pub fn check(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let now_ms = unix_ms();
        let mut entry = self
            .buckets
            .entry(ip)
            .or_insert_with(|| TokenBucket::new(self.capacity, self.refill_per_ms, now, now_ms));
        entry.try_consume(now, now_ms)
    }

    pub fn evict_stale(&self, older_than: Duration) {
        let cutoff = unix_ms().saturating_sub(older_than.as_millis() as u64);
        self.buckets.retain(|_, bucket| bucket.last_seen_ms >= cutoff);
    }

    #[cfg(test)]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.buckets.len()
    }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn extract_client_ip(req: &Request) -> Option<IpAddr> {
    let headers = req.headers();
    if let Some(value) = headers.get("x-forwarded-for")
        && let Ok(s) = value.to_str()
    {
        let first = s.split(',').next().unwrap_or("").trim();
        if let Ok(ip) = first.parse::<IpAddr>() {
            return Some(ip);
        }
    }
    if let Some(value) = headers.get("x-real-ip")
        && let Ok(s) = value.to_str()
        && let Ok(ip) = s.trim().parse::<IpAddr>()
    {
        return Some(ip);
    }
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip())
}

pub async fn rate_limit_middleware(
    State(limiter): State<Arc<RateLimiter>>,
    req: Request,
    next: Next,
) -> Response {
    let ip = extract_client_ip(&req).unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
    if limiter.check(ip) {
        next.run(req).await
    } else {
        tracing::debug!(client_ip = %ip, "rate limit rejected request");
        (
            StatusCode::TOO_MANY_REQUESTS,
            [(axum::http::header::RETRY_AFTER, "60")],
            Json(serde_json::json!({
                "error": "Too many requests",
                "code": "rate_limited",
            })),
        )
            .into_response()
    }
}
