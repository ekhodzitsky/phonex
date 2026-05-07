//! Model management: auto-download, extraction, and verification.

use futures_util::StreamExt;
use std::io::Write;
use std::path::Path;

/// Specification for a downloadable Sherpa-ONNX model.
pub struct ModelSpec {
    pub dir_name: &'static str,
    pub archive_name: &'static str,
    pub url: &'static str,
    pub sha256: Option<&'static str>,
}

/// Registry of known models that can be auto-downloaded.
pub const KNOWN_MODELS: &[ModelSpec] = &[
    // Offline models
    ModelSpec {
        dir_name: "sherpa-onnx-small-zipformer-ru-2024-09-18",
        archive_name: "sherpa-onnx-small-zipformer-ru-2024-09-18.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-small-zipformer-ru-2024-09-18.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
        archive_name: "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        archive_name: "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-gigaspeech-2023-12-12",
        archive_name: "sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-en-2023-06-26",
        archive_name: "sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-en-libriheavy-20230926-small",
        archive_name: "sherpa-onnx-zipformer-en-libriheavy-20230926-small.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-libriheavy-20230926-small.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-thai-2024-06-20",
        archive_name: "sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-cantonese-2024-03-13",
        archive_name: "sherpa-onnx-zipformer-cantonese-2024-03-13.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-cantonese-2024-03-13.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-zh-en-2023-11-22",
        archive_name: "sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-zipformer-korean-2024-06-24",
        archive_name: "sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2",
        sha256: None,
    },
    // Streaming models
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-en-2023-06-21",
        archive_name: "sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        archive_name: "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-fr-2023-04-14",
        archive_name: "sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06",
        archive_name: "sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06",
        archive_name: "sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-korean-2024-06-16",
        archive_name: "sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        archive_name: "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2",
        sha256: None,
    },
    ModelSpec {
        dir_name: "sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09",
        archive_name: "sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09.tar.bz2",
        sha256: None,
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

    if let Some(expected) = spec.sha256 {
        let actual = sha256_file(&archive_path)
            .map_err(|e| crate::SiamError::Inference(format!("Checksum computation failed: {e}")))?;
        if actual != expected {
            return Err(crate::SiamError::Inference(format!(
                "Checksum mismatch for {}: expected {expected}, got {actual}",
                spec.archive_name
            )));
        }
        tracing::info!(archive = %archive_path.display(), "Checksum verified");
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

    for entry in archive.entries()? {
        let mut entry = entry.map_err(|e| crate::SiamError::Inference(format!("Extraction failed: {e}")))?;
        // unpack_in validates that the entry path stays within dest
        entry.unpack_in(dest)
            .map_err(|e| crate::SiamError::Inference(format!("Extraction failed: {e}")))?;
    }
    Ok(())
}

/// Compute the SHA-256 hex digest of a file.
fn sha256_file(path: &Path) -> crate::Result<String> {
    use sha2::Digest;
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let result = hasher.finalize();
    Ok(result.iter().map(|b| format!("{:02x}", b)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn make_malicious_tar_bz2(path: &Path) {
        // Build a raw ustar header for "../etc/passwd" to bypass tar::Builder
        // path validation.
        let mut header = [0u8; 512];
        let name = b"../etc/passwd";
        header[..name.len()].copy_from_slice(name);

        let mode = b"0000644 ";
        header[100..108].copy_from_slice(mode);
        let uid = b"0001750 ";
        header[108..116].copy_from_slice(uid);
        let gid = b"0001750 ";
        header[116..124].copy_from_slice(gid);
        let size = b"00000000000 ";
        header[124..136].copy_from_slice(size);
        let mtime = b"00000000000 ";
        header[136..148].copy_from_slice(mtime);

        // checksum placeholder (8 spaces)
        header[148..156].copy_from_slice(b"        ");
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");

        let sum: u64 = header.iter().map(|&b| b as u64).sum();
        let cksum = format!("{:06o} \0", sum);
        header[148..156].copy_from_slice(cksum.as_bytes());

        let file = std::fs::File::create(path).unwrap();
        let mut encoder =
            bzip2::write::BzEncoder::new(file, bzip2::Compression::default());
        encoder.write_all(&header).unwrap();
        // Two zero-blocks mark end-of-archive
        encoder.write_all(&[0u8; 1024]).unwrap();
        encoder.finish().unwrap();
    }

    #[test]
    fn test_extract_tar_bz2_rejects_path_traversal() {
        let dir = tempdir().unwrap();
        let archive_path = dir.path().join("malicious.tar.bz2");
        let dest = dir.path().join("dest");
        std::fs::create_dir(&dest).unwrap();

        make_malicious_tar_bz2(&archive_path);

        // extract_tar_bz2 should either fail or silently skip the entry;
        // in either case the file must NOT be created outside dest.
        let _ = extract_tar_bz2(&archive_path, &dest);

        let escaped_path = dir.path().join("etc").join("passwd");
        assert!(
            !escaped_path.exists(),
            "path traversal entry must not be extracted outside destination"
        );
    }

    #[test]
    fn test_sha256_file_matches_expected() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");

        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(b"hello world").unwrap();
        drop(file);

        let result = sha256_file(&file_path).unwrap();
        let expected =
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert_eq!(result, expected);
    }
}
