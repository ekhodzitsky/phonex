//! Streaming inference state.

use serde::Serialize;
use std::sync::Arc;

/// A recognized word with timing and confidence metadata.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct WordInfo {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
}

/// Result of file transcription, including word-level details.
#[derive(Debug, Clone, Serialize)]
pub struct TranscribeResult {
    pub text: String,
    pub words: Vec<WordInfo>,
    pub duration_s: f64,
}

/// A transcript segment emitted by the inference engine.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct TranscriptSegment {
    #[serde(skip)]
    pub text: Arc<String>,
    #[serde(skip)]
    pub words: Arc<Vec<WordInfo>>,
    pub is_final: bool,
    pub timestamp: f64,
}

impl TranscriptSegment {
    /// Return the text as a plain String for serialization helpers.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Return the words slice for serialization helpers.
    pub fn words(&self) -> &[WordInfo] {
        &self.words
    }
}

/// Per-connection streaming state that persists across audio chunks.
#[non_exhaustive]
pub struct StreamingState {
    /// Accumulated audio samples for the growing-buffer model.
    pub audio_buffer: Vec<f32>,
    /// Number of samples to keep as overlap after each window.
    pub overlap_samples: usize,
    /// Number of samples required to trigger a partial transcription.
    pub window_samples: usize,
}

impl Default for StreamingState {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingState {
    pub fn new() -> Self {
        Self {
            audio_buffer: Vec::new(),
            overlap_samples: (super::TARGET_SAMPLE_RATE as f64 * 2.0) as usize,
            window_samples: (super::TARGET_SAMPLE_RATE as f64 * 5.0) as usize,
        }
    }

    pub fn clear(&mut self) {
        self.audio_buffer.clear();
    }

    /// Returns true if the buffer has reached the window threshold.
    pub fn should_process(&self) -> bool {
        self.audio_buffer.len() >= self.window_samples
    }

    /// Trim the buffer, keeping only the overlap tail.
    pub fn trim_overlap(&mut self) {
        if self.audio_buffer.len() > self.overlap_samples {
            let start = self.audio_buffer.len() - self.overlap_samples;
            let tail = self.audio_buffer.split_off(start);
            self.audio_buffer = tail;
        }
    }
}

pub fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
