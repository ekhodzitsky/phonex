//! Speaker diarization integration via polyvoice.

use std::path::Path;

use polyvoice::{DiarizationConfig, OfflineDiarizer, OnnxEmbeddingExtractor, SpeakerTurn};

/// Engine for speaker diarization.
pub struct DiarizationEngine {
    diarizer: OfflineDiarizer,
    extractor: OnnxEmbeddingExtractor,
}

impl DiarizationEngine {
    /// Load a diarization engine from an ONNX speaker embedding model.
    ///
    /// # Arguments
    /// * `model_path` — path to the ONNX speaker embedding model (e.g. WeSpeaker ResNet34)
    /// * `embedding_dim` — output dimension of the embedding model (e.g. 256)
    /// * `window_samples` — analysis window size in samples (e.g. 24000 for 1.5s @ 16kHz)
    /// * `pool_size` — number of parallel ONNX sessions
    pub fn new(
        model_path: &Path,
        embedding_dim: usize,
        window_samples: usize,
        pool_size: usize,
    ) -> anyhow::Result<Self> {
        let extractor = OnnxEmbeddingExtractor::new(model_path, embedding_dim, window_samples, pool_size)?;
        let diarizer = OfflineDiarizer::new(DiarizationConfig::default());
        Ok(Self { diarizer, extractor })
    }

    /// Run diarization on a mono f32 audio buffer at 16 kHz.
    pub fn diarize(&self, samples: &[f32]) -> anyhow::Result<Vec<SpeakerTurn>> {
        let result = self.diarizer.run(samples, &self.extractor)?;
        Ok(result.turns)
    }
}

/// Assign speaker IDs to transcribed words based on diarization turns.
pub fn assign_speakers(
    words: &mut [crate::inference::WordInfo],
    turns: &[SpeakerTurn],
) {
    for word in words.iter_mut() {
        let word_mid = (word.start + word.end) / 2.0;
        for turn in turns {
            if word_mid >= turn.time.start && word_mid <= turn.time.end {
                word.speaker = Some(turn.speaker.0 as u32);
                break;
            }
        }
    }
}
