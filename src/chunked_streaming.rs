//! Pseudo-streaming pipeline for offline models using VAD-triggered segmentation.
//!
//! Accumulates audio, runs Silero VAD in real-time, and triggers offline
//! transcription on each speech→silence transition. Works with any offline
//! model, giving ~500–800 ms end-of-utterance latency.

use std::sync::Arc;

use crate::inference::Engine;
use crate::streaming_decoder::DecodeToken;
use crate::vad::StreamingVad;

/// A chunked streaming pipeline that wraps an offline `Engine`.
///
/// Unlike `StreamingPipeline` (which needs a streaming Zipformer model),
/// this works with any offline model by buffering speech segments and
/// running full offline transcription at utterance boundaries.
pub struct ChunkedStreamingPipeline {
    engine: Arc<Engine>,
    vad: StreamingVad,
    /// Samples accumulated during the current speech segment.
    speech_buffer: Vec<f32>,
    /// Global time offset in seconds for the current utterance.
    global_time_offset: f64,
    /// All tokens produced so far.
    all_tokens: Vec<DecodeToken>,
}

impl ChunkedStreamingPipeline {
    /// Create a new chunked pipeline from an engine and a VAD model path.
    pub fn new(engine: Arc<Engine>, vad_path: &str) -> crate::Result<Self> {
        let vad = StreamingVad::new(vad_path)?;
        Ok(Self {
            engine,
            vad,
            speech_buffer: Vec::new(),
            global_time_offset: 0.0,
            all_tokens: Vec::new(),
        })
    }

    /// Feed audio samples into the pipeline.
    ///
    /// Returns newly decoded tokens whenever a speech segment ends.
    pub async fn accept_audio(&mut self, samples: &[f32]) -> crate::Result<Vec<DecodeToken>> {
        const MAX_SPEECH_BUFFER_SECONDS: f64 = 30.0;
        const MAX_SPEECH_BUFFER_SAMPLES: usize = (MAX_SPEECH_BUFFER_SECONDS * 16000.0) as usize;

        let (speech_samples, speech_ended) = self.vad.process_with_transitions(samples);
        self.speech_buffer.extend_from_slice(&speech_samples);

        let force_transcribe = self.speech_buffer.len() > MAX_SPEECH_BUFFER_SAMPLES;
        if force_transcribe {
            tracing::warn!("Speech buffer exceeded {}s, forcing transcription", MAX_SPEECH_BUFFER_SECONDS);
        }

        if (speech_ended || force_transcribe) && !self.speech_buffer.is_empty() {
            self.transcribe_buffer().await
        } else {
            Ok(Vec::new())
        }
    }

    /// Finalize and return the full transcript.
    pub async fn flush(&mut self) -> crate::Result<String> {
        if !self.speech_buffer.is_empty() {
            let _ = self.transcribe_buffer().await?;
        }
        Ok(self.text())
    }

    /// Return accumulated text so far.
    pub fn text(&self) -> String {
        // Concatenate token texts.
        self.all_tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().concat()
    }

    async fn transcribe_buffer(&mut self) -> crate::Result<Vec<DecodeToken>> {
        let engine = self.engine.clone();
        let samples = std::mem::take(&mut self.speech_buffer);
        let guard = self.engine.pool.checkout().await
            .map_err(|_| crate::SiamError::Inference("Pool empty".into()))?;
        let guard = guard.into_owned();
        let offset = self.global_time_offset;

        let result = tokio::task::spawn_blocking(move || {
            let mut guard = guard;
            engine.transcribe_samples(&samples, &mut *guard)
        })
        .await
        .map_err(|e| crate::SiamError::Inference(format!("spawn_blocking error: {e}")))?;

        let result = match result {
            Ok(r) => r,
            Err(e) => return Err(e),
        };

        let mut new_tokens = Vec::new();
        for word in result.words {
            new_tokens.push(DecodeToken {
                id: 0,
                text: word.word,
                start: offset + word.start,
                end: offset + word.end,
                confidence: word.confidence,
            });
        }

        self.global_time_offset += result.duration_s;
        self.all_tokens.extend(new_tokens.clone());
        Ok(new_tokens)
    }
}
