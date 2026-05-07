//! ONNX session loader with platform-specific execution providers.

use ort::session::Session;

/// Load an ONNX model with the best available execution provider.
///
/// On Apple platforms with the `coreml` feature enabled, uses CoreML with
/// Neural Engine. Falls back to CPU-only otherwise.
pub fn load_onnx_session(path: &str) -> crate::Result<Session> {
    load_onnx_session_impl(path, false)
}

/// Load an ONNX model on CPU only (no CoreML / GPU).
pub fn load_onnx_session_cpu(path: &str) -> crate::Result<Session> {
    load_onnx_session_impl(path, true)
}

fn ort_err(msg: &str, e: impl std::fmt::Display) -> crate::SiamError {
    crate::SiamError::Inference(format!("{msg}: {e}"))
}

fn load_onnx_session_impl(path: &str, #[allow(unused_variables)] force_cpu: bool) -> crate::Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| ort_err("ORT session builder error", e))?
        .with_intra_threads(4)
        .map_err(|e| ort_err("ORT session builder error", e))?;

    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    if !force_cpu {
        let ep = ort::ep::CoreML::default()
            .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine)
            .with_specialization_strategy(ort::ep::coreml::SpecializationStrategy::FastPrediction)
            .build();
        builder = builder
            .with_execution_providers([ep])
            .map_err(|e| ort_err("ORT execution provider error", e))?;
        let session = builder
            .commit_from_file(path)
            .map_err(|e| ort_err("ORT commit error", e))?;
        tracing::info!(model = %path, "Loaded ONNX session with CoreML (experimental — may be slower than CPU for some models)");
        return Ok(session);
    }

    #[cfg(feature = "cuda")]
    if !force_cpu {
        let ep = ort::ep::CUDA::default().build();
        builder = builder
            .with_execution_providers([ep])
            .map_err(|e| ort_err("ORT execution provider error", e))?;
        let session = builder
            .commit_from_file(path)
            .map_err(|e| ort_err("ORT commit error", e))?;
        tracing::info!(model = %path, "Loaded ONNX session with CUDA");
        return Ok(session);
    }

    builder
        .commit_from_file(path)
        .map_err(|e| ort_err("ORT commit error", e))
}
