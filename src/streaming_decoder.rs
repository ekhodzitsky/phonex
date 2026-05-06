//! Stateful greedy RNNT decoder for streaming inference.

use ndarray::{Array2, Array3};

use crate::decoder::SherpaDecoder;
use crate::joiner::SherpaJoiner;
use crate::inference::decode::argmax_logit_with_confidence;
use crate::tokenizer::Tokenizer;

/// A single decoded token with timing and confidence.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecodeToken {
    pub id: u32,
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f32,
}

/// Greedy RNNT decoder that maintains state across chunks.
pub struct StreamingGreedyDecoder<'a> {
    decoder: &'a mut SherpaDecoder,
    joiner: &'a mut SherpaJoiner,
    tokenizer: &'a Tokenizer,
    blank_id: u32,
    d_model: usize,
    context_size: usize,
    context: Array2<i64>,
    global_time_offset: f64,
}

impl<'a> StreamingGreedyDecoder<'a> {
    pub fn new(
        decoder: &'a mut SherpaDecoder,
        joiner: &'a mut SherpaJoiner,
        tokenizer: &'a Tokenizer,
        context_size: usize,
    ) -> Self {
        let blank_id = tokenizer.blank_id();
        let d_model = decoder.d_model();
        let context = Array2::from_elem((1, context_size), blank_id as i64);
        Self {
            decoder,
            joiner,
            tokenizer,
            blank_id,
            d_model,
            context_size,
            context,
            global_time_offset: 0.0,
        }
    }

    /// Decode a chunk of encoder output. Returns newly emitted tokens.
    pub fn decode_chunk(&mut self, encoder_out: &Array3<f32>) -> crate::Result<Vec<DecodeToken>> {
        let time = encoder_out.shape()[1];
        let mut new_tokens = Vec::new();

        for t in 0..time {
            let mut enc_frame = Array2::zeros((1, self.d_model));
            for ch in 0..self.d_model {
                enc_frame[[0, ch]] = encoder_out[[0, t, ch]];
            }

            let mut tokens_this_step = 0;
            loop {
                if tokens_this_step >= crate::inference::MAX_TOKENS_PER_STEP {
                    break;
                }

                let decoder_out = self.decoder.step(&self.context)?;
                let logits = self.joiner.step(&enc_frame, &decoder_out)?;
                let (pred_id, confidence) = argmax_logit_with_confidence(&logits, self.blank_id);

                if pred_id == self.blank_id {
                    break;
                }

                for i in 0..(self.context_size - 1) {
                    self.context[[0, i]] = self.context[[0, i + 1]];
                }
                self.context[[0, self.context_size - 1]] = pred_id as i64;

                let abs_t = self.global_time_offset + t as f64 * crate::inference::FRAME_SHIFT_S;
                new_tokens.push(DecodeToken {
                    id: pred_id,
                    text: String::new(),
                    start: abs_t,
                    end: abs_t + crate::inference::FRAME_SHIFT_S,
                    confidence,
                });
                tokens_this_step += 1;
            }
        }

        self.global_time_offset += time as f64 * crate::inference::FRAME_SHIFT_S;

        // Fill token texts
        for token in &mut new_tokens {
            token.text = self.tokenizer.decode_ids(&[token.id]);
        }

        // Merge end times
        for i in 0..new_tokens.len().saturating_sub(1) {
            new_tokens[i].end = new_tokens[i + 1].start;
        }

        Ok(new_tokens)
    }

    /// Reset decoder state for a new utterance.
    pub fn reset(&mut self) {
        self.context = Array2::from_elem((1, self.context_size), self.blank_id as i64);
        self.global_time_offset = 0.0;
    }

    /// Decode accumulated token ids to full text.
    pub fn decode_text(&self, tokens: &[DecodeToken]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        self.tokenizer.decode_ids(&ids)
    }
}


