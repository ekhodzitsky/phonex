//! Model management: auto-download, extraction, and verification.

use futures_util::StreamExt;
use std::io::Write;
use std::path::Path;

/// Specification for a downloadable Sherpa-ONNX model.
pub struct ModelSpec {
    pub dir_name: &'static str,
    pub archive_name: &'static str,
    pub url: &'static str,
}

/// Registry of known models that can be auto-downloaded.
pub const KNOWN_MODELS: &[ModelSpec] = &[
    // Offline models
    ModelSpec {
        dir_name: "sherpa-onnx-small-zipformer-ru-2024-09-18",
        archive_name: "sherpa-onnx-small-zipformer-ru-2024-09-18.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-small-zipformer-ru-2024-09-18.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
        archive_name: "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        archive_name: "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-gigaspeech-2023-12-12",
        archive_name: "sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-en-2023-06-26",
        archive_name: "sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-en-libriheavy-20230926-small",
        archive_name: "sherpa-onnx-zipformer-en-libriheavy-20230926-small.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-libriheavy-20230926-small.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-thai-2024-06-20",
        archive_name: "sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-cantonese-2024-03-13",
        archive_name: "sherpa-onnx-zipformer-cantonese-2024-03-13.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-cantonese-2024-03-13.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-zh-en-2023-11-22",
        archive_name: "sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-korean-2024-06-24",
        archive_name: "sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2",
    },
    // Streaming models
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-en-2023-06-21",
        archive_name: "sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        archive_name: "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-fr-2023-04-14",
        archive_name: "sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06",
        archive_name: "sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06",
        archive_name: "sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-korean-2024-06-16",
        archive_name: "sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        archive_name: "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2",
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09",
        archive_name: "sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09.tar.bz2",
    },
];

/// Find a model spec by directory name.
pub fn find_model_spec(dir_name: &str) -> Option<&'static ModelSpec> {
    KNOWN_MODELS.iter().find(|m| m.dir_name == dir_name)
}

/// Check if all required model files exist in the given directory.
pub fn model_exists(model_dir: &str) -> bool {
    crate::model_config::discover_model_files(model_dir).is_ok()
}

/// Ensure the model is available, downloading it if necessary.
///
/// If the model directory basename matches a known model in [`KNOWN_MODELS`],
/// it will be auto-downloaded from the official Sherpa-ONNX release.
pub fn ensure_model(model_dir: &str) -> crate::Result<()> {
    if model_exists(model_dir) {
        tracing::debug!(model_dir, "Model already present");
        return Ok(());
    }

    let basename = Path::new(model_dir)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model_dir);

    let spec = find_model_spec(basename)
        .ok_or_else(|| crate::SiamError::Inference(
            format!("Unknown model '{}'. Please download it manually or use a supported --language.", model_dir)
        ))?;

    let models_base = Path::new(model_dir)
        .parent()
        .unwrap_or(Path::new("models"));
    std::fs::create_dir_all(models_base)?;

    let archive_path = models_base.join(spec.archive_name);

    if !archive_path.exists() {
        tracing::info!(url = spec.url, "Downloading model archive");
        download_with_progress(spec.url, &archive_path)?;
    }

    tracing::info!(archive = %archive_path.display(), "Extracting model archive");
    extract_tar_bz2(&archive_path, models_base)?;

    // Move extracted directory to the expected location if needed
    let extracted_dir = models_base.join(spec.dir_name);
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
