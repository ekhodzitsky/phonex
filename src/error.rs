use thiserror::Error;

#[derive(Error, Debug)]
pub enum SiamError {
    #[error("ONNX error: {0}")]
    Onnx(#[from] ort::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Audio error: {0}")]
    Audio(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("NDArray shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
}

pub type Result<T> = std::result::Result<T, SiamError>;
