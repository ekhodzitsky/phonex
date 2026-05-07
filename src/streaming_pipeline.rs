//! End-to-end streaming inference pipeline.

use std::sync::Arc;

use ndarray::{Array2, Array3};

use crate::decoder::SherpaDecoder;
use crate::inference::decode::argmax_logit_with_confidence;
use crate::inference::features::phonex_fbank_options;
use crate::joiner::SherpaJoiner;
use crate::model_config::{ModelInfo, discover_model_files};
use crate::streaming_decoder::DecodeToken;
use crate::streaming_encoder::StreamingEncoder;
use crate::tokenizer::Tokenizer;
use crate::vad::StreamingVad;

use kaldi_native_fbank::fbank::FbankComputer;
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};

const SAMPLE_RATE: f32 = 16000.0;

/// End-to-end streaming ASR pipeline (encoder + decoder + feature extraction).
pub struct StreamingPipeline {
    encoder: StreamingEncoder,
    decoder: SherpaDecoder,
    joiner: SherpaJoiner,
    tokenizer: Arc<Tokenizer>,
    online: OnlineFeature,
    n_mels: usize,
    context_size: usize,
    chunk_frames: usize,
    chunk_shift: usize,
    next_frame_idx: usize,
    all_tokens: Vec<DecodeToken>,
    decoder_context: Array2<i64>,
    global_time_offset: f64,
    blank_id: u32,
    d_model: usize,
    vad: Option<StreamingVad>,
    last_final_text: Option<String>,
}

impl StreamingPipeline {
    pub fn from_model_dir(
        model_dir: &str,
        info: &ModelInfo,
        vad_path: Option<&str>,
    ) -> crate::Result<Self> {
        let paths = discover_model_files(model_dir)?;

        let encoder = StreamingEncoder::new(paths.encoder.to_str().unwrap_or(""))?;
        let chunk_frames = encoder.chunk_frames();
        let chunk_shift = encoder.chunk_shift();
        let decoder = SherpaDecoder::new(paths.decoder.to_str().unwrap_or(""), info)?;
        let joiner = SherpaJoiner::new(paths.joiner.to_str().unwrap_or(""), info)?;
        let tokenizer = Arc::new(Tokenizer::from_file(
            paths.tokenizer.to_str().unwrap_or(""),
            paths.tokens.to_str().unwrap_or(""),
            info.blank_id,
        )?);

        let d_model = decoder.d_model();

        let opts = phonex_fbank_options();
        let computer = FbankComputer::new(opts.clone()).expect("FBANK options valid");
        let n_mels = computer.dim();
        let online = OnlineFeature::new(FeatureComputer::Fbank(computer));

        let context = Array2::from_elem((1, info.context_size), info.blank_id as i64);

        let vad = if let Some(path) = vad_path {
            Some(StreamingVad::new(path)?)
        } else {
            None
        };

        Ok(Self {
            encoder,
            decoder,
            joiner,
            tokenizer,
            online,
            n_mels,
            context_size: info.context_size,
            chunk_frames,
            chunk_shift,
            next_frame_idx: 0,
            all_tokens: Vec::new(),
            decoder_context: context,
            global_time_offset: 0.0,
            blank_id: info.blank_id,
            d_model,
            vad,
            last_final_text: None,
        })
    }

    /// Feed raw audio samples and return any newly emitted tokens.
    pub fn accept_audio(&mut self, samples: &[f32]) -> crate::Result<Vec<DecodeToken>> {
        if let Some(vad) = &mut self.vad {
            let was_speech = vad.is_speech();
            let speech_samples = vad.process(samples);
            let is_speech = vad.is_speech();

            if !speech_samples.is_empty() {
                self.online.accept_waveform(SAMPLE_RATE, &speech_samples);
            }

            let tokens = self.process_ready_frames()?;

            if was_speech && !is_speech {
                let text = self.flush()?;
                self.last_final_text = Some(text);
                self.reset();
            }

            Ok(tokens)
        } else {
            self.online.accept_waveform(SAMPLE_RATE, samples);
            self.process_ready_frames()
        }
    }

    fn process_ready_frames(&mut self) -> crate::Result<Vec<DecodeToken>> {
        let mut new_tokens = Vec::new();
        while self.online.num_frames_ready() >= self.next_frame_idx + self.chunk_frames {
            let chunk = self.extract_chunk(self.next_frame_idx, self.chunk_frames);
            self.next_frame_idx += self.chunk_shift;

            let x = Array3::from_shape_vec((1, self.chunk_frames, self.n_mels), chunk)
                .map_err(crate::SiamError::Shape)?;

            let encoder_out = self.encoder.encode_chunk(&x)?;
            let chunk_tokens = self.decode_chunk(&encoder_out)?;
            new_tokens.extend(chunk_tokens);
        }
        self.all_tokens.extend(new_tokens.clone());
        Ok(new_tokens)
    }

    fn extract_chunk(&self, start: usize, num_frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; num_frames * self.n_mels];
        for f in 0..num_frames {
            let frame = self
                .online
                .get_frame(start + f)
                .expect("frame index < num_frames_ready");
            out[f * self.n_mels..(f + 1) * self.n_mels].copy_from_slice(&frame[..self.n_mels]);
        }
        out
    }

    fn decode_chunk(&mut self, encoder_out: &Array3<f32>) -> crate::Result<Vec<DecodeToken>> {
        let time = encoder_out.shape()[1];
        let mut chunk_tokens = Vec::new();

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

                let decoder_out = self.decoder.step(&self.decoder_context)?;
                let logits = self.joiner.step(&enc_frame, &decoder_out)?;
                let (pred_id, confidence) = argmax_logit_with_confidence(&logits, self.blank_id);

                if pred_id == self.blank_id {
                    break;
                }

                for i in 0..(self.context_size - 1) {
                    self.decoder_context[[0, i]] = self.decoder_context[[0, i + 1]];
                }
                self.decoder_context[[0, self.context_size - 1]] = pred_id as i64;

                let abs_t = self.global_time_offset + t as f64 * crate::inference::FRAME_SHIFT_S;
                chunk_tokens.push(DecodeToken {
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

        for token in &mut chunk_tokens {
            token.text = self.tokenizer.decode_ids(&[token.id]);
        }
        for i in 0..chunk_tokens.len().saturating_sub(1) {
            chunk_tokens[i].end = chunk_tokens[i + 1].start;
        }

        Ok(chunk_tokens)
    }

    /// Finalize decoding, process any trailing frames, and return full text.
    pub fn flush(&mut self) -> crate::Result<String> {
        let (text, _tokens) = self.flush_with_tokens()?;
        Ok(text)
    }

    /// Finalize decoding and return both text and word-level timestamps.
    pub fn flush_with_tokens(&mut self) -> crate::Result<(String, Vec<DecodeToken>)> {
        self.online.input_finished();
        let remaining = self
            .online
            .num_frames_ready()
            .saturating_sub(self.next_frame_idx);
        if remaining > 0 {
            let chunk = self.extract_chunk(self.next_frame_idx, remaining);
            let mut padded = vec![0.0f32; self.chunk_frames * self.n_mels];
            padded[..chunk.len()].copy_from_slice(&chunk);

            let x = Array3::from_shape_vec((1, self.chunk_frames, self.n_mels), padded)
                .map_err(crate::SiamError::Shape)?;

            let encoder_out = self.encoder.encode_chunk(&x)?;
            let chunk_tokens = self.decode_chunk(&encoder_out)?;
            self.all_tokens.extend(chunk_tokens);
        }

        let ids: Vec<u32> = self.all_tokens.iter().map(|t| t.id).collect();
        let text = self.tokenizer.decode_ids(&ids);
        Ok((text, self.all_tokens.clone()))
    }

    /// Return accumulated tokens so far.
    pub fn tokens(&self) -> &[DecodeToken] {
        &self.all_tokens
    }

    /// Return the properly decoded accumulated text so far.
    pub fn text(&self) -> String {
        let ids: Vec<u32> = self.all_tokens.iter().map(|t| t.id).collect();
        self.tokenizer.decode_ids(&ids)
    }

    /// Reset the pipeline for a new utterance.
    pub fn reset(&mut self) {
        self.encoder.reset();
        self.decoder_context = Array2::from_elem((1, self.context_size), self.blank_id as i64);
        self.global_time_offset = 0.0;
        self.next_frame_idx = 0;
        self.all_tokens.clear();
        let opts = phonex_fbank_options();
        let computer = FbankComputer::new(opts).expect("FBANK options valid");
        self.online = OnlineFeature::new(FeatureComputer::Fbank(computer));
        if let Some(vad) = &mut self.vad {
            vad.reset();
        }
    }

    /// Returns `true` if the VAD is currently in the `Speech` state.
    pub fn has_active_speech(&self) -> bool {
        self.vad.as_ref().is_some_and(|v| v.is_speech())
    }

    /// Take the final text produced by a VAD-triggered flush, if any.
    pub fn take_final_text(&mut self) -> Option<String> {
        self.last_final_text.take()
    }
}

#[cfg(test)]
mod tests {
    fn _assert_send<T: Send>() {}
    fn _assert_send_pipeline() {
        _assert_send::<super::StreamingPipeline>();
    }
}
