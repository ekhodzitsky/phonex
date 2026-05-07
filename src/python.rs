//! PyO3 Python bindings for the phonex STT engine.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Python wrapper around the Rust inference [`Engine`].
#[pyclass(unsendable)]
pub struct PhonexEngine {
    engine: crate::Engine,
}

#[pymethods]
impl PhonexEngine {
    #[new]
    fn new(model_dir: &str) -> PyResult<Self> {
        let engine = crate::Engine::load(model_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load engine: {e}")))?;
        Ok(Self { engine })
    }

    /// Transcribe an audio file and return the text.
    fn transcribe(&self, audio_path: &str) -> PyResult<String> {
        self.engine
            .transcribe_file(audio_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Transcription failed: {e}")))
    }

    /// Transcribe raw audio samples (f32 PCM at the model sample rate) and return the text.
    fn transcribe_samples(&self, samples: Vec<f32>) -> PyResult<String> {
        let mut guard = self
            .engine
            .pool
            .try_checkout()
            .ok_or_else(|| PyRuntimeError::new_err("Session pool exhausted"))?;
        let result = self
            .engine
            .transcribe_samples(&samples, &mut guard)
            .map_err(|e| PyRuntimeError::new_err(format!("Transcription failed: {e}")))?;
        Ok(result.text)
    }
}

/// The `phonex` Python extension module.
#[pymodule]
fn phonex(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PhonexEngine>()?;
    Ok(())
}
