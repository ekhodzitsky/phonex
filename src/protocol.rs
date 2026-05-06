//! WebSocket protocol messages for phonex.

use serde::{Deserialize, Serialize};

/// Current WebSocket protocol version.
pub const PROTOCOL_VERSION: &str = "1.0";

/// Server → Client messages.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ServerMessage {
    /// Server is ready to accept audio.
    Ready {
        model: String,
        sample_rate: u32,
        version: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        supported_rates: Vec<u32>,
    },
    /// Partial (interim) transcript.
    Partial {
        text: String,
        timestamp: f64,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        words: Vec<crate::inference::WordInfo>,
    },
    /// Final transcript.
    Final {
        text: String,
        timestamp: f64,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        words: Vec<crate::inference::WordInfo>,
    },
    /// Error occurred during processing.
    Error {
        message: String,
        code: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u32>,
    },
}

/// Client → Server text messages.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ClientMessage {
    /// Request server to stop and finalize.
    Stop,
    /// Clear accumulated audio and reset session.
    Clear,
    /// Configure session parameters.
    Configure {
        #[serde(default)]
        sample_rate: Option<u32>,
    },
}
