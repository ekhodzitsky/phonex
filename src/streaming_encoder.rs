//! Streaming Zipformer encoder with stateful cache management.

use std::collections::HashMap;

use ndarray::Array3;
use ort::session::Session;
use ort::value::Tensor;

/// Stateful streaming encoder for Sherpa-ONNX Zipformer models.
///
/// The encoder expects fixed-size chunks (e.g. 39 frames) and maintains
/// internal caches (cached_len, cached_avg, cached_key, cached_val,
/// cached_val2, cached_conv1, cached_conv2) between calls.
pub struct StreamingEncoder {
    session: Session,
    states: HashMap<String, ort::value::Value>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl StreamingEncoder {
    pub fn new(model_path: &str) -> crate::Result<Self> {
        // NOTE: CoreML works with the streaming encoder (verified 2026-05-05),
        // but for this specific model it is ~6x slower than CPU
        // (≈1.0 s vs ≈0.17 s per 5-second audio clip on M-series Mac).
        // Keeping CPU-only as the default until CoreML performance improves.
        let session = crate::session::load_onnx_session_cpu(model_path)?;
        Self::from_session(session)
    }

    fn zero_state_tensor(name: &str, shape: Vec<usize>) -> crate::Result<ort::value::Value> {
        let value = if name.starts_with("cached_len_") {
            Tensor::from_array(ndarray::Array::<i64, _>::zeros(shape))?.into_dyn()
        } else {
            Tensor::from_array(ndarray::Array::<f32, _>::zeros(shape))?.into_dyn()
        };
        Ok(value)
    }

    pub fn from_session(session: Session) -> crate::Result<Self> {
        let mut states = HashMap::new();

        for input in session.inputs() {
            let name = input.name().to_string();
            if !name.starts_with("cached_") {
                continue;
            }
            let shape: Vec<usize> = match input.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => shape
                    .iter()
                    .map(|&d| if d <= 0 { 1usize } else { d as usize })
                    .collect(),
                _ => continue,
            };

            states.insert(name.clone(), Self::zero_state_tensor(&name, shape)?);
        }

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        Ok(Self {
            session,
            states,
            input_names,
            output_names,
        })
    }

    /// Encode a single chunk. `x` shape: `[batch, chunk_frames, n_mels]`.
    pub fn encode_chunk(&mut self, x: &Array3<f32>) -> crate::Result<Array3<f32>> {
        let mut x_owned = Some(Tensor::from_array(x.to_owned())?);
        let mut inputs: Vec<(
            std::borrow::Cow<'_, str>,
            ort::session::SessionInputValue<'_>,
        )> = Vec::with_capacity(self.input_names.len());

        for name in &self.input_names {
            if name == "x" {
                let x_tensor = x_owned.take().ok_or_else(|| {
                    crate::SiamError::Inference("Encoder input 'x' missing".into())
                })?;
                inputs.push((std::borrow::Cow::Borrowed("x"), x_tensor.into()));
            } else if name.starts_with("cached_") {
                let state = self.states.get(name.as_str()).ok_or_else(|| {
                    crate::SiamError::Inference(format!("Missing encoder state: {}", name))
                })?;
                inputs.push((std::borrow::Cow::Borrowed(name.as_str()), state.into()));
            } else {
                return Err(crate::SiamError::Inference(format!(
                    "Unknown encoder input: {}",
                    name
                )));
            }
        }

        let outputs = self.session.run(inputs)?;

        let encoder_out = outputs["encoder_out"]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality()?;

        // Update states from new_cached_* outputs
        for name in &self.output_names {
            if !name.starts_with("new_cached_") {
                continue;
            }
            let input_name = name.replacen("new_", "", 1);
            let new_value = match outputs[name.as_str()].dtype() {
                ort::value::ValueType::Tensor { ty, .. } => match ty {
                    ort::value::TensorElementType::Int64 => {
                        let arr = outputs[name.as_str()]
                            .try_extract_array::<i64>()?
                            .to_owned();
                        Tensor::from_array(arr)?.into_dyn()
                    }
                    ort::value::TensorElementType::Float32 => {
                        let arr = outputs[name.as_str()]
                            .try_extract_array::<f32>()?
                            .to_owned();
                        Tensor::from_array(arr)?.into_dyn()
                    }
                    _ => continue,
                },
                _ => continue,
            };
            self.states.insert(input_name, new_value);
        }

        Ok(encoder_out)
    }

    /// Return the expected number of frames per chunk (from ONNX input shape).
    pub fn chunk_frames(&self) -> usize {
        self.session
            .inputs()
            .iter()
            .find(|i| i.name() == "x")
            .and_then(|i| match i.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => {
                    let dim = shape.get(1).copied().unwrap_or(39);
                    Some(if dim <= 0 { 39 } else { dim as usize })
                }
                _ => None,
            })
            .unwrap_or(39)
    }

    /// Return the frame shift between chunks (from ONNX metadata).
    pub fn chunk_shift(&self) -> usize {
        self.session
            .metadata()
            .ok()
            .and_then(|m| m.custom("decode_chunk_len"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(32)
    }

    /// Reset all cached states to zeros (e.g. on new utterance).
    pub fn reset(&mut self) {
        for input in self.session.inputs() {
            let name = input.name().to_string();
            if !name.starts_with("cached_") {
                continue;
            }
            let shape: Vec<usize> = match input.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => shape
                    .iter()
                    .map(|&d| if d <= 0 { 1usize } else { d as usize })
                    .collect(),
                _ => continue,
            };

            self.states.insert(
                name.clone(),
                match Self::zero_state_tensor(&name, shape) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!("Failed to create zero state tensor: {e}");
                        continue;
                    }
                },
            );
        }
    }
}
