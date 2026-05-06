use ndarray::Array2;
use ort::value::TensorRef;

use crate::model_config::ModelInfo;

/// Sherpa-ONNX decoder (prediction network).
///
/// Input:  y [batch, context_size]
/// Output: decoder_out [batch, d_model]
pub struct SherpaDecoder {
    session: ort::session::Session,
    d_model: usize,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl SherpaDecoder {
    pub fn new(model_path: &str, info: &ModelInfo) -> crate::Result<Self> {
        let session = crate::session::load_onnx_session(model_path)?;
        Ok(Self {
            session,
            d_model: info.d_model,
            input_names: info.decoder_inputs.clone(),
            output_names: info.decoder_outputs.clone(),
        })
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn step(&mut self, y: &Array2<i64>) -> crate::Result<Array2<f32>> {
        let y_tensor = TensorRef::from_array_view(y.view())?;

        let outputs = self.session.run(ort::inputs![
            self.input_names[0].as_str() => y_tensor,
        ])?;

        let decoder_out = outputs[self.output_names[0].as_str()]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality()?;

        Ok(decoder_out)
    }
}
