use thiserror::Error;

/// Unified error type for the phonex STT pipeline.
#[derive(Error, Debug)]
pub enum SiamError {
    /// ONNX Runtime session or inference failure.
    #[error("ONNX error: {0}")]
    Onnx(#[from] ort::Error),

    /// File system or I/O failure.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// SentencePiece tokenizer initialization or encoding error.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Audio decoding, resampling, or format error.
    #[error("Audio error: {0}")]
    Audio(String),

    /// Generic inference pipeline failure (e.g., invalid model inputs).
    #[error("Inference error: {0}")]
    Inference(String),

    /// NDArray shape mismatch during tensor construction.
    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    /// NDArray shape mismatch with explicit expected vs actual dimensions.
    #[error("NDArray shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape dimensions.
        expected: Vec<usize>,
        /// Actual shape dimensions received.
        got: Vec<usize>,
    },
}

/// Convenience alias for [`std::result::Result<T, SiamError>`].
pub type Result<T> = std::result::Result<T, SiamError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let onnx = SiamError::Onnx(ort::Error::new("test onnx error"));
        assert!(onnx.to_string().contains("ONNX error"));

        let io = SiamError::Io(std::io::Error::new(std::io::ErrorKind::Other, "test io"));
        assert!(io.to_string().contains("IO error"));

        let tok = SiamError::Tokenizer("bad token".into());
        assert!(tok.to_string().contains("Tokenizer error"));

        let audio = SiamError::Audio("bad audio".into());
        assert!(audio.to_string().contains("Audio error"));

        let inf = SiamError::Inference("bad inference".into());
        assert!(inf.to_string().contains("Inference error"));

        let shape = SiamError::Shape(ndarray::ShapeError::from_kind(
            ndarray::ErrorKind::OutOfBounds,
        ));
        assert!(shape.to_string().contains("Shape error"));

        let mismatch = SiamError::ShapeMismatch {
            expected: vec![1, 2],
            got: vec![3, 4],
        };
        assert!(mismatch.to_string().contains("NDArray shape mismatch"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let siam: SiamError = io_err.into();
        assert!(matches!(siam, SiamError::Io(_)));
        assert!(siam.to_string().contains("file not found"));
    }
}
