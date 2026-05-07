//! Model configuration: auto-detect parameters from ONNX session shapes.

use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Discovered file paths for a Sherpa-ONNX model directory.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokenizer: PathBuf,
    pub tokens: PathBuf,
    pub vad: Option<PathBuf>,
}

/// Discover model files in a directory using loose naming conventions.
///
/// Supports any Sherpa-ONNX release layout:
/// - `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`
/// - `bpe.model`, `tokenizer.model`, or any `*.model`
/// - `tokens.txt`
/// - `silero_vad.onnx` (optional)
pub fn discover_model_files(model_dir: &str) -> crate::Result<ModelPaths> {
    let dir = Path::new(model_dir);
    if !dir.is_dir() {
        return Err(crate::SiamError::Inference(format!("Not a directory: {}", model_dir)));
    }

    let mut encoder = None;
    let mut decoder = None;
    let mut joiner = None;
    let mut tokenizer = None;
    let mut tokens = None;
    let mut vad = None;

    for entry in std::fs::read_dir(dir).map_err(crate::SiamError::Io)? {
        let entry = entry.map_err(crate::SiamError::Io)?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let lower = name.to_lowercase();

        if lower.starts_with("encoder") && lower.ends_with(".onnx") {
            encoder = pick_priority(encoder, entry.path(), &lower);
        } else if lower.starts_with("decoder") && lower.ends_with(".onnx") {
            decoder = pick_priority(decoder, entry.path(), &lower);
        } else if lower.starts_with("joiner") && lower.ends_with(".onnx") {
            joiner = pick_priority(joiner, entry.path(), &lower);
        } else if lower == "tokens.txt" {
            tokens = Some(entry.path());
        } else if lower.ends_with(".model") {
            // Prefer bpe.model / tokenizer.model over random .model files
            tokenizer = pick_tokenizer(tokenizer, entry.path(), &lower);
        } else if lower == "silero_vad.onnx" {
            vad = Some(entry.path());
        }
    }

    let require = |name: &str, p: Option<PathBuf>| -> crate::Result<PathBuf> {
        p.ok_or_else(|| crate::SiamError::Inference(
            format!("Model directory '{}' missing {} file", model_dir, name)
        ))
    };

    Ok(ModelPaths {
        encoder: require("encoder*.onnx", encoder)?,
        decoder: require("decoder*.onnx", decoder)?,
        joiner: require("joiner*.onnx", joiner)?,
        tokenizer: tokenizer.unwrap_or_default(), // may be empty if only tokens.txt is used
        tokens: require("tokens.txt", tokens)?,
        vad,
    })
}

/// Pick between two candidate files, preferring int8 quantized variants.
fn pick_priority(current: Option<PathBuf>, candidate: PathBuf, lower: &str) -> Option<PathBuf> {
    match current {
        None => Some(candidate),
        Some(ref existing) if lower.contains("int8") && !existing.to_string_lossy().to_lowercase().contains("int8") => Some(candidate),
        Some(existing) => Some(existing),
    }
}

/// Pick tokenizer model, preferring bpe.model / tokenizer.model.
fn pick_tokenizer(current: Option<PathBuf>, candidate: PathBuf, lower: &str) -> Option<PathBuf> {
    match current {
        None => Some(candidate),
        Some(ref existing) if lower.contains("bpe") || lower.contains("tokenizer") => Some(candidate),
        Some(existing) => Some(existing),
    }
}

/// User-provided model configuration (optional JSON file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,
    #[serde(default = "default_blank_id")]
    pub blank_id: u32,
    #[serde(default)]
    pub context_size: Option<usize>,
    #[serde(default)]
    pub d_model: Option<usize>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub encoder: Option<TensorConfig>,
    #[serde(default)]
    pub decoder: Option<TensorConfig>,
    #[serde(default)]
    pub joiner: Option<TensorConfig>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            n_mels: default_n_mels(),
            blank_id: default_blank_id(),
            context_size: None,
            d_model: None,
            vocab_size: None,
            encoder: None,
            decoder: None,
            joiner: None,
            model_id: None,
            model_name: None,
        }
    }
}

fn default_sample_rate() -> u32 { 16000 }
fn default_n_mels() -> usize { 80 }
fn default_blank_id() -> u32 { 0 }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TensorConfig {
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

/// Auto-detected model parameters from ONNX sessions.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub sample_rate: u32,
    pub n_mels: usize,
    pub blank_id: u32,
    pub context_size: usize,
    pub d_model: usize,
    pub vocab_size: usize,
    pub encoder_inputs: Vec<String>,
    pub encoder_outputs: Vec<String>,
    pub decoder_inputs: Vec<String>,
    pub decoder_outputs: Vec<String>,
    pub joiner_inputs: Vec<String>,
    pub joiner_outputs: Vec<String>,
    pub model_id: String,
    pub model_name: String,
}

impl ModelInfo {
    /// Returns `true` if the model is a streaming (online) model.
    /// Streaming models have `cached_*` inputs in the encoder ONNX graph.
    pub fn is_streaming(&self) -> bool {
        self.encoder_inputs.iter().any(|name| name.starts_with("cached_"))
    }
}

trait Named {
    fn name(&self) -> &str;
}

impl Named for ort::value::Outlet {
    fn name(&self) -> &str {
        self.name()
    }
}

fn io_names(config_names: Option<Vec<String>>, fallback: &[impl Named]) -> Vec<String> {
    config_names.unwrap_or_else(|| fallback.iter().map(|v| v.name().to_string()).collect())
}

impl ModelInfo {
    /// Load model info from ONNX sessions, optionally overridden by a JSON config.
    pub fn from_model_dir(model_dir: &str) -> crate::Result<Self> {
        let config_path = Path::new(model_dir).join("model.json");
        let config: ModelConfig = if config_path.exists() {
            let text = std::fs::read_to_string(&config_path)
                .map_err(crate::SiamError::Io)?;
            serde_json::from_str(&text)
                .map_err(|e| crate::SiamError::Inference(format!("Invalid model.json: {e}")))?
        } else {
            ModelConfig::default()
        };

        let paths = discover_model_files(model_dir)?;

        // Use probe sessions to read shapes. We load them temporarily and extract metadata.
        let encoder = Session::builder()
            .map_err(|e| crate::SiamError::Inference(format!("ORT builder: {e}")))?
            .commit_from_file(&paths.encoder)?;
        let decoder = Session::builder()
            .map_err(|e| crate::SiamError::Inference(format!("ORT builder: {e}")))?
            .commit_from_file(&paths.decoder)?;
        let joiner = Session::builder()
            .map_err(|e| crate::SiamError::Inference(format!("ORT builder: {e}")))?
            .commit_from_file(&paths.joiner)?;

        let n_mels = config.n_mels;
        let d_model = config.d_model.unwrap_or_else(|| {
            // Try to infer from encoder output shape [batch, time, d_model]
            encoder.outputs().iter().find_map(|o| {
                if let ort::value::ValueType::Tensor { shape, .. } = o.dtype()
                    && shape.len() == 3 {
                        return Some(shape[2] as usize);
                    }
                None
            }).unwrap_or(512)
        });

        let context_size = config.context_size.unwrap_or_else(|| {
            decoder.inputs().iter().find_map(|i| {
                if let ort::value::ValueType::Tensor { shape, .. } = i.dtype()
                    && shape.len() == 2 {
                        return Some(shape[1] as usize);
                    }
                None
            }).unwrap_or(2)
        });

        let vocab_size = config.vocab_size.unwrap_or_else(|| {
            joiner.outputs().iter().find_map(|o| {
                if let ort::value::ValueType::Tensor { shape, .. } = o.dtype()
                    && shape.len() == 2 {
                        return Some(shape[1] as usize);
                    }
                None
            }).unwrap_or(2000)
        });

        let encoder_inputs = io_names(config.encoder.as_ref().map(|c| c.input_names.clone()), encoder.inputs());
        let encoder_outputs = io_names(config.encoder.as_ref().map(|c| c.output_names.clone()), encoder.outputs());
        let decoder_inputs = io_names(config.decoder.as_ref().map(|c| c.input_names.clone()), decoder.inputs());
        let decoder_outputs = io_names(config.decoder.as_ref().map(|c| c.output_names.clone()), decoder.outputs());
        let joiner_inputs = io_names(config.joiner.as_ref().map(|c| c.input_names.clone()), joiner.inputs());
        let joiner_outputs = io_names(config.joiner.as_ref().map(|c| c.output_names.clone()), joiner.outputs());

        // Derive model_id from config or directory basename.
        let dir_basename = Path::new(model_dir)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".into());
        let model_id = config.model_id.clone().unwrap_or_else(|| dir_basename.clone());
        let model_name = config.model_name.clone().unwrap_or(dir_basename);

        // If there was no model.json, write one with auto-detected values so the
        // next startup skips ONNX probing.
        if !config_path.exists() {
            let auto_config = ModelConfig {
                sample_rate: config.sample_rate,
                n_mels,
                blank_id: config.blank_id,
                context_size: Some(context_size),
                d_model: Some(d_model),
                vocab_size: Some(vocab_size),
                encoder: Some(TensorConfig {
                    input_names: encoder_inputs.clone(),
                    output_names: encoder_outputs.clone(),
                }),
                decoder: Some(TensorConfig {
                    input_names: decoder_inputs.clone(),
                    output_names: decoder_outputs.clone(),
                }),
                joiner: Some(TensorConfig {
                    input_names: joiner_inputs.clone(),
                    output_names: joiner_outputs.clone(),
                }),
                model_id: Some(model_id.clone()),
                model_name: Some(model_name.clone()),
            };
            if let Ok(json) = serde_json::to_string_pretty(&auto_config) {
                let _ = std::fs::write(&config_path, json);
                tracing::info!(path = %config_path.display(), "Wrote auto-detected model.json");
            }
        }

        Ok(Self {
            sample_rate: config.sample_rate,
            n_mels,
            blank_id: config.blank_id,
            context_size,
            d_model,
            vocab_size,
            encoder_inputs,
            encoder_outputs,
            decoder_inputs,
            decoder_outputs,
            joiner_inputs,
            joiner_outputs,
            model_id,
            model_name,
        })
    }
}
