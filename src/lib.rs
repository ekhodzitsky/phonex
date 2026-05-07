//! # phonex
//!
//! Generic on-device speech-to-text powered by Sherpa-ONNX Zipformer via ONNX Runtime.

pub mod audio;
pub mod encoder;
pub mod decoder;
pub mod error;
pub mod inference;
pub mod joiner;
pub mod model;
pub mod model_config;
#[cfg(feature = "server")]
pub mod protocol;
#[cfg(feature = "server")]
pub mod server;
pub mod session;
pub mod streaming_decoder;
pub mod streaming_encoder;
pub mod streaming_pipeline;
pub mod tokenizer;
pub mod vad;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use error::{Result, SiamError};
pub use inference::Engine;
pub use streaming_decoder::DecodeToken;
pub use streaming_encoder::StreamingEncoder;
pub use streaming_pipeline::StreamingPipeline;
