//! Inference pipeline: audio preprocessing → feature extraction → encoder → decoder → joiner.

pub mod audio;
pub mod decode;
pub mod engine;
pub mod features;
pub mod pool;
pub mod streaming;
pub mod tokenizer;

/// Target sample rate for the model (Hz).
pub const TARGET_SAMPLE_RATE: u32 = 16000;

/// Supported input sample rates (Hz).
pub const SUPPORTED_RATES: &[u32] = &[8000, 16000, 24000, 44100, 48000];

/// Number of mel bins expected by the encoder.
pub const N_MELS: usize = 80;

/// Maximum tokens emitted for a single encoder frame before forced blank.
pub const MAX_TOKENS_PER_STEP: usize = 20;

/// Frame shift in seconds for kaldi-native-fbank (10 ms).
pub const FRAME_SHIFT_S: f64 = 0.01;

pub use audio::resample;
pub use engine::Engine;
pub use pool::{OwnedReservation, PoolGuard, SessionPool, SessionTriplet};
pub use streaming::{TranscribeResult, TranscriptSegment, WordInfo};
