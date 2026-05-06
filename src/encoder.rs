use ndarray::{Array1, Array3};
use ort::value::TensorRef;

use crate::model_config::ModelInfo;

/// Offline Zipformer encoder (Sherpa-ONNX style).
///
/// Input:  mel [batch, time, n_mels]  (time-major)
/// Output: encoder_out [batch, time, d_model]
pub struct OfflineEncoder {
    session: ort::session::Session,
    d_model: usize,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OfflineEncoder {
    pub fn new(model_path: &str, info: &ModelInfo) -> crate::Result<Self> {
        let session = crate::session::load_onnx_session(model_path)?;
        Ok(Self {
            session,
            d_model: info.d_model,
            input_names: info.encoder_inputs.clone(),
            output_names: info.encoder_outputs.clone(),
        })
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn encode(
        &mut self,
        mel: &Array3<f32>,
        lengths: &Array1<i64>,
    ) -> crate::Result<(Array3<f32>, Array1<i64>)> {
        let mel_tensor = TensorRef::from_array_view(mel.view())?;
        let len_tensor = TensorRef::from_array_view(lengths.view())?;

        let outputs = self.session.run(ort::inputs![
            self.input_names[0].as_str() => mel_tensor,
            self.input_names[1].as_str() => len_tensor,
        ])?;

        let encoded = outputs[self.output_names[0].as_str()]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality()?;
        let encoded_lengths = outputs[self.output_names[1].as_str()]
            .try_extract_array::<i64>()?
            .to_owned()
            .into_dimensionality()?;

        Ok((encoded, encoded_lengths))
    }
}
