//! Model manifest: reproducible model downloads with SHA-256 pinning.

use serde::Deserialize;
use std::path::Path;
use std::sync::OnceLock;

/// A single entry in `models/manifest.json` describing a downloadable model.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelManifest {
    pub name: String,
    pub url: String,
    pub sha256: Option<String>,
    pub size_bytes: u64,
    pub language: String,
    #[serde(rename = "model_type")]
    pub model_type: String,
}

/// Wrapper so the top-level JSON object can hold an array under the key `models`.
#[derive(Debug, Clone, Deserialize)]
struct ManifestFile {
    models: Vec<ModelManifest>,
}

/// Load `models/manifest.json` and return the list of model entries.
///
/// Returns an error if the file is missing or malformed.
pub fn load_manifest() -> crate::Result<Vec<ModelManifest>> {
    let path = Path::new("models/manifest.json");
    let text = std::fs::read_to_string(path).map_err(|e| {
        crate::SiamError::Inference(format!(
            "Failed to read model manifest at {}: {e}",
            path.display()
        ))
    })?;
    let manifest: ManifestFile = serde_json::from_str(&text)
        .map_err(|e| crate::SiamError::Inference(format!("Failed to parse model manifest: {e}")))?;
    Ok(manifest.models)
}

/// Find a manifest entry whose `url` matches the given value.
pub fn find_manifest_by_url(url: &str) -> Option<ModelManifest> {
    static MANIFEST: OnceLock<Vec<ModelManifest>> = OnceLock::new();
    let entries = MANIFEST.get_or_init(|| load_manifest().unwrap_or_default());
    entries.iter().find(|m| m.url == url).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_manifest_parses() {
        let manifest = load_manifest();
        assert!(
            manifest.is_ok(),
            "manifest.json should load without error: {:?}",
            manifest.err()
        );
        let entries = manifest.unwrap();
        assert!(
            !entries.is_empty(),
            "manifest should contain at least one entry"
        );
    }

    #[test]
    fn test_find_manifest_by_url_known() {
        let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2";
        let entry = find_manifest_by_url(url);
        assert!(entry.is_some(), "should find English offline model by URL");
        let entry = entry.unwrap();
        assert_eq!(entry.language, "english");
        assert_eq!(entry.model_type, "offline");
    }

    #[test]
    fn test_find_manifest_by_url_unknown() {
        let entry = find_manifest_by_url("https://example.com/unknown.tar.bz2");
        assert!(entry.is_none(), "unknown URL should return None");
    }
}
