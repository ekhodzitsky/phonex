//! Model management: auto-download, extraction, and verification.

use futures_util::StreamExt;
use std::io::Write;
use std::path::Path;

const MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2";

/// Check if all required model files exist in the given directory.
pub fn model_exists(model_dir: &str) -> bool {
    crate::model_config::discover_model_files(model_dir).is_ok()
}

/// Ensure the model is available, downloading it if necessary.
pub fn ensure_model(model_dir: &str) -> crate::Result<()> {
    if model_exists(model_dir) {
        tracing::debug!(model_dir, "Model already present");
        return Ok(());
    }

    let models_base = Path::new(model_dir)
        .parent()
        .unwrap_or(Path::new("models"));
    std::fs::create_dir_all(models_base)?;

    let archive_path = models_base.join("sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2");

    if !archive_path.exists() {
        tracing::info!(url = MODEL_URL, "Downloading model archive");
        download_with_progress(MODEL_URL, &archive_path)?;
    }

    tracing::info!(archive = %archive_path.display(), "Extracting model archive");
    extract_tar_bz2(&archive_path, models_base)?;

    // Move extracted directory to the expected location if needed
    let extracted_dir = models_base.join("sherpa-onnx-zipformer-thai-2024-06-20");
    let target_dir = Path::new(model_dir);
    if extracted_dir != target_dir && extracted_dir.exists() {
        if target_dir.exists() {
            std::fs::remove_dir_all(target_dir)?;
        }
        std::fs::rename(&extracted_dir, target_dir)?;
    }

    // Copy VAD model from parent models/ dir if it exists there but not in model dir
    let parent_vad = models_base.join("silero_vad.onnx");
    let target_vad = target_dir.join("silero_vad.onnx");
    if parent_vad.exists() && !target_vad.exists() {
        std::fs::copy(&parent_vad, &target_vad)?;
    }

    if !model_exists(model_dir) {
        return Err(crate::SiamError::Inference(
            format!("Model extraction failed: missing files in {}", model_dir)
        ));
    }

    tracing::info!(model_dir, "Model ready");
    Ok(())
}

/// Download a file with a progress bar.
fn download_with_progress(url: &str, dest: &Path) -> crate::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async_download(url, dest))
}

async fn async_download(url: &str, dest: &Path) -> crate::Result<()> {
    let response = reqwest::get(url).await
        .map_err(|e| crate::SiamError::Inference(format!("Download failed: {e}")))?;

    let total_size = response.content_length().unwrap_or(0);
    let pb = indicatif::ProgressBar::new(total_size);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message("Downloading model");

    let mut file = std::fs::File::create(dest)?;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| crate::SiamError::Inference(format!("Download stream error: {e}")))?;
        file.write_all(&chunk)?;
        pb.inc(chunk.len() as u64);
    }

    pb.finish_with_message("Download complete");
    Ok(())
}

/// Extract a tar.bz2 archive.
fn extract_tar_bz2(archive: &Path, dest: &Path) -> crate::Result<()> {
    let file = std::fs::File::open(archive)?;
    let decompressor = bzip2::read::BzDecoder::new(file);
    let mut archive = tar::Archive::new(decompressor);
    archive.unpack(dest)
        .map_err(|e| crate::SiamError::Inference(format!("Extraction failed: {e}")))?;
    Ok(())
}
