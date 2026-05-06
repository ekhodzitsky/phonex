use ndarray::Array2;
use ort::value::TensorRef;

use crate::model_config::ModelInfo;

/// Sherpa-ONNX joiner network.
///
/// Input:  encoder_out [batch, d_model], decoder_out [batch, d_model]
/// Output: logit [batch, vocab_size]
pub struct SherpaJoiner {
    session: ort::session::Session,
    vocab_size: usize,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl SherpaJoiner {
    pub fn new(model_path: &str, info: &ModelInfo) -> crate::Result<Self> {
        let session = crate::session::load_onnx_session(model_path)?;
        Ok(Self {
            session,
            vocab_size: info.vocab_size,
            input_names: info.joiner_inputs.clone(),
            output_names: info.joiner_outputs.clone(),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn step(
        &mut self,
        encoder_out: &Array2<f32>,
        decoder_out: &Array2<f32>,
    ) -> crate::Result<Array2<f32>> {
        let enc_tensor = TensorRef::from_array_view(encoder_out.view())?;
        let dec_tensor = TensorRef::from_array_view(decoder_out.view())?;

        let outputs = self.session.run(ort::inputs![
            self.input_names[0].as_str() => enc_tensor,
            self.input_names[1].as_str() => dec_tensor,
        ])?;

        let logits = outputs[self.output_names[0].as_str()]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality()?;

        Ok(logits)
    }
}
