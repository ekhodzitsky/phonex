use ndarray::{Array1, Array2, Array3, Axis};

use crate::decoder::SherpaDecoder;
use crate::encoder::OfflineEncoder;
use crate::joiner::SherpaJoiner;
use crate::tokenizer::Tokenizer;

/// A single decoded token with timing and confidence.
#[derive(Debug, Clone)]
pub struct DecodeToken {
    pub id: u32,
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f32,
}

/// Greedy RNNT decoder for Sherpa-ONNX offline Zipformer models.
pub struct GreedyDecoder<'a> {
    encoder: &'a mut OfflineEncoder,
    decoder: &'a mut SherpaDecoder,
    joiner: &'a mut SherpaJoiner,
    tokenizer: &'a Tokenizer,
    blank_id: u32,
    d_model: usize,
    context_size: usize,
}

impl<'a> GreedyDecoder<'a> {
    pub fn new(
        encoder: &'a mut OfflineEncoder,
        decoder: &'a mut SherpaDecoder,
        joiner: &'a mut SherpaJoiner,
        tokenizer: &'a Tokenizer,
        context_size: usize,
    ) -> Self {
        let blank_id = tokenizer.blank_id();
        let d_model = encoder.d_model();
        Self {
            encoder,
            decoder,
            joiner,
            tokenizer,
            blank_id,
            d_model,
            context_size,
        }
    }

    /// Transcribe a single mel spectrogram utterance.
    ///
    /// `mel` must be [batch, time, n_mels] (time-major).
    pub fn transcribe_offline(
        &mut self,
        mel: &Array3<f32>,
        _audio_length: usize,
    ) -> crate::Result<String> {
        let (text, _) = self.transcribe_offline_with_tokens(mel)?;
        Ok(text)
    }

    /// Transcribe and return both the full text and per-token details.
    pub fn transcribe_offline_with_tokens(
        &mut self,
        mel: &Array3<f32>,
    ) -> crate::Result<(String, Vec<DecodeToken>)> {
        let valid_frames = mel.shape()[1] as i64;
        let lengths = Array1::from_vec(vec![valid_frames]);
        let (encoded, enc_lengths) = self.encoder.encode(mel, &lengths)?;
        let max_t = enc_lengths[0] as usize;
        self.decode_sample(&encoded, 0, max_t)
    }

    /// Decode a single sample from a batched encoder output.
    pub fn decode_sample(
        &mut self,
        encoded: &Array3<f32>,
        sample_idx: usize,
        max_t: usize,
    ) -> crate::Result<(String, Vec<DecodeToken>)> {
        let mut context = Array2::from_elem((1, self.context_size), i64::from(self.blank_id));
        let mut tokens: Vec<DecodeToken> = Vec::new();

        for t in 0..max_t {
            let mut enc_frame = Array2::zeros((1, self.d_model));
            for ch in 0..self.d_model {
                enc_frame[[0, ch]] = encoded[[sample_idx, t, ch]];
            }

            let mut tokens_this_step = 0;
            loop {
                if tokens_this_step >= super::MAX_TOKENS_PER_STEP {
                    break;
                }

                let decoder_out = self.decoder.step(&context)?;
                let logits = self.joiner.step(&enc_frame, &decoder_out)?;
                let (pred_id, confidence) = argmax_logit_with_confidence(&logits, self.blank_id);

                if pred_id == self.blank_id {
                    break;
                }

                for i in 0..(self.context_size - 1) {
                    context[[0, i]] = context[[0, i + 1]];
                }
                context[[0, self.context_size - 1]] = i64::from(pred_id);

                tokens.push(DecodeToken {
                    id: pred_id,
                    text: String::new(),
                    start: t as f64 * super::FRAME_SHIFT_S,
                    end: (t + 1) as f64 * super::FRAME_SHIFT_S,
                    confidence,
                });
                tokens_this_step += 1;
            }
        }

        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        let text = self.tokenizer.decode_ids(&ids);

        for token in &mut tokens {
            token.text = self.tokenizer.decode_ids(&[token.id]);
        }

        for i in 0..tokens.len().saturating_sub(1) {
            tokens[i].end = tokens[i + 1].start;
        }
        if let Some(last) = tokens.last_mut() {
            last.end = max_t as f64 * super::FRAME_SHIFT_S;
        }

        Ok((text, tokens))
    }
}

/// Argmax over logits [batch, vocab_size], returning the predicted token id and softmax confidence.
pub fn argmax_logit_with_confidence(logits: &Array2<f32>, blank_id: u32) -> (u32, f32) {
    let view = logits.index_axis(Axis(0), 0);
    let slice: Vec<f32> = view.iter().copied().collect();

    let max_logit = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = slice.iter().map(|&x| (x - max_logit).exp()).sum();

    let (idx, max_val) = slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or((blank_id as usize, 0.0), |(idx, &val)| (idx, val));

    let confidence = if exp_sum > 0.0 {
        (max_val - max_logit).exp() / exp_sum
    } else {
        1.0
    };

    (idx as u32, confidence)
}
