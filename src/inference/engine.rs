//! Core ONNX inference engine.

use bytes::Bytes;
use ndarray::{Array1, Array3};
use std::sync::Arc;

use super::decode::GreedyDecoder;
use super::features::MelSpectrogram;
use super::pool::{SessionPool, SessionTriplet};
use super::streaming::{StreamingState, TranscribeResult, TranscriptSegment};
use crate::error::SiamError;
use crate::model_config::ModelInfo;
use crate::tokenizer::Tokenizer;
use crate::vad::{Vad, VadConfig};

pub struct Engine {
    pub pool: SessionPool,
    pub mel: MelSpectrogram,
    pub tokenizer: Arc<Tokenizer>,
    pub vocab_size: usize,
    pub info: ModelInfo,
    pub vad: Option<Vad>,
    #[cfg(feature = "diarization")]
    pub diarization: Option<crate::diarization::DiarizationEngine>,
}

impl Engine {
    /// Load an engine from a model directory (CLI convenience).
    pub fn load(model_dir: &str) -> crate::Result<Self> {
        Self::load_with_pool_size(model_dir, 1)
    }

    /// Load an engine from a model directory with a custom pool size.
    pub fn load_with_pool_size(model_dir: &str, pool_size: usize) -> crate::Result<Self> {
        let info = ModelInfo::from_model_dir(model_dir)?;
        let paths = crate::model_config::discover_model_files(model_dir)?;
        let tokenizer = Arc::new(Tokenizer::from_file(
            paths.tokenizer.to_str().unwrap_or(""),
            paths.tokens.to_str().unwrap_or(""),
            info.blank_id,
        )?);
        let mut triplets = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            triplets.push(SessionTriplet::from_model_dir(model_dir, &info)?);
        }
        let pool = SessionPool::new(triplets);
        Ok(Self::new(pool, tokenizer, info))
    }

    /// Build an engine with the given session pool, tokenizer, and model info.
    pub fn new(pool: SessionPool, tokenizer: Arc<Tokenizer>, info: ModelInfo) -> Self {
        let vocab_size = tokenizer.vocab_size();
        Self {
            pool,
            mel: MelSpectrogram::new(),
            tokenizer,
            vocab_size,
            info,
            vad: None,
            #[cfg(feature = "diarization")]
            diarization: None,
        }
    }

    /// Build a test-only engine with an empty pool.
    #[cfg(test)]
    pub fn test_stub() -> Self {
        let info = ModelInfo::from_model_dir("models/sherpa-onnx-zipformer-thai-2024-06-20")
            .unwrap_or_else(|_| ModelInfo {
                sample_rate: 16000,
                n_mels: 80,
                blank_id: 0,
                context_size: 2,
                d_model: 512,
                vocab_size: 2000,
                encoder_inputs: vec!["x".into(), "x_lens".into()],
                encoder_outputs: vec!["encoder_out".into(), "encoder_out_lens".into()],
                decoder_inputs: vec!["y".into()],
                decoder_outputs: vec!["decoder_out".into()],
                joiner_inputs: vec!["encoder_out".into(), "decoder_out".into()],
                joiner_outputs: vec!["logit".into()],
                model_id: "sherpa-onnx-zipformer-thai-2024-06-20".into(),
                model_name: "Sherpa-ONNX Zipformer".into(),
            });
        Self {
            pool: SessionPool::new(vec![]),
            mel: MelSpectrogram::new(),
            tokenizer: Arc::new(Tokenizer::from_file("", "", 0).unwrap()),
            vocab_size: 2000,
            info,
            vad: None,
            #[cfg(feature = "diarization")]
            diarization: None,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn with_vad(mut self, model_path: &str) -> Self {
        if std::path::Path::new(model_path).exists() {
            match Vad::new(VadConfig {
                model_path: model_path.into(),
                ..VadConfig::default()
            }) {
                Ok(vad) => self.vad = Some(vad),
                Err(e) => tracing::warn!("Failed to load VAD model: {e}"),
            }
        }
        self
    }

    #[cfg(feature = "diarization")]
    pub fn with_diarization(mut self, model_path: &str) -> Self {
        let path = std::path::Path::new(model_path);
        if path.exists() {
            match crate::diarization::DiarizationEngine::new(path, 256, 24000, 1) {
                Ok(engine) => {
                    tracing::info!(model = %model_path, "Loaded diarization engine");
                    self.diarization = Some(engine);
                }
                Err(e) => {
                    tracing::warn!(model = %model_path, "Failed to load diarization engine: {e}");
                }
            }
        }
        self
    }

    pub fn transcribe_bytes_shared(
        &self,
        data: &Bytes,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, SiamError> {
        let samples = crate::inference::audio::bytes_to_f32_samples(data);
        self.transcribe_samples(&samples, triplet)
    }

    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, SiamError> {
        let duration_s = samples.len() as f64 / self.info.sample_rate as f64;

        let (features_flat, num_frames) = self.mel.compute(samples);
        let mel = Array3::from_shape_vec((1, num_frames, self.info.n_mels), features_flat)
            .map_err(SiamError::Shape)?;

        let mut decoder = GreedyDecoder::new(
            &mut triplet.encoder,
            &mut triplet.decoder,
            &mut triplet.joiner,
            &self.tokenizer,
            self.info.context_size,
        );

        let (text, tokens) = decoder.transcribe_offline_with_tokens(&mel)?;

        let words = tokens
            .into_iter()
            .map(|t| super::streaming::WordInfo {
                word: t.text,
                start: t.start,
                end: t.end.min(duration_s),
                confidence: t.confidence,
                speaker: None,
            })
            .collect();

        Ok(TranscribeResult {
            text,
            words,
            duration_s,
        })
    }

    /// Transcribe an audio file synchronously (CLI convenience).
    pub fn transcribe_file(&self, path: &str) -> crate::Result<String> {
        let result = self.transcribe_file_with_details(path)?;
        Ok(result.text)
    }

    /// Transcribe an audio file and return full details (text + words with timing).
    pub fn transcribe_file_with_details(&self, path: &str) -> crate::Result<TranscribeResult> {
        let (samples, sample_rate) = crate::audio::AudioPreprocessor::read_wav(path)?;
        let samples = if sample_rate == self.info.sample_rate as usize {
            samples
        } else {
            crate::audio::AudioPreprocessor::typhoon().resample(&samples, sample_rate)?
        };
        let mut guard = self
            .pool
            .try_checkout()
            .ok_or_else(|| SiamError::Inference("Pool empty".into()))?;
        self.transcribe_samples(&samples, &mut guard)
    }

    /// Transcribe a batch of audio samples efficiently using batched encoder.
    pub fn transcribe_batch(
        &self,
        samples: Vec<&[f32]>,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscribeResult>, SiamError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        if samples.len() == 1 {
            return Ok(vec![self.transcribe_samples(samples[0], triplet)?]);
        }

        // 1. Compute mel for each sample
        let mut mels = Vec::with_capacity(samples.len());
        let mut max_frames = 0;
        for s in &samples {
            let (features_flat, num_frames) = self.mel.compute(s);
            max_frames = max_frames.max(num_frames);
            mels.push((features_flat, num_frames, s.len()));
        }

        // 2. Pad and build batch tensor [batch, max_frames, n_mels]
        let batch_size = samples.len();
        let mut batch_data = vec![0.0f32; batch_size * max_frames * self.info.n_mels];
        let mut lengths_vec = Vec::with_capacity(batch_size);

        for (b, (features_flat, num_frames, _sample_len)) in mels.into_iter().enumerate() {
            lengths_vec.push(num_frames as i64);
            for f in 0..num_frames {
                for m in 0..self.info.n_mels {
                    let src_idx = f * self.info.n_mels + m;
                    let dst_idx = b * max_frames * self.info.n_mels + f * self.info.n_mels + m;
                    batch_data[dst_idx] = features_flat[src_idx];
                }
            }
        }

        let batch_mel =
            Array3::from_shape_vec((batch_size, max_frames, self.info.n_mels), batch_data)
                .map_err(SiamError::Shape)?;
        let lengths = Array1::from_vec(lengths_vec);

        // 3. Batch encode
        let (encoded, enc_lengths) = triplet.encoder.encode(&batch_mel, &lengths)?;

        // 4. Decode each sample individually
        let mut decoder = GreedyDecoder::new(
            &mut triplet.encoder,
            &mut triplet.decoder,
            &mut triplet.joiner,
            &self.tokenizer,
            self.info.context_size,
        );

        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let max_t = enc_lengths[b] as usize;
            let duration_s = samples[b].len() as f64 / self.info.sample_rate as f64;
            let (text, tokens) = decoder.decode_sample(&encoded, b, max_t)?;

            let words = tokens
                .into_iter()
                .map(|t| super::streaming::WordInfo {
                    word: t.text,
                    start: t.start,
                    end: t.end.min(duration_s),
                    confidence: t.confidence,
                    speaker: None,
                })
                .collect();

            results.push(TranscribeResult {
                text,
                words,
                duration_s,
            });
        }

        Ok(results)
    }

    pub fn transcribe_samples_with_vad(
        &self,
        samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, SiamError> {
        let mut vad = Vad::new(VadConfig {
            model_path: "models/silero_vad.onnx".into(),
            ..VadConfig::default()
        })
        .map_err(|e| SiamError::Inference(format!("Failed to initialize VAD: {e}")))?;
        let segments = vad.split(samples, self.info.sample_rate as usize);

        let mut all_words = Vec::new();
        let mut full_text = String::new();

        for (speaker_id, (start, end)) in segments.iter().enumerate() {
            let chunk = &samples[*start..*end];
            let result = self.transcribe_samples(chunk, triplet)?;

            let offset_sec = *start as f64 / self.info.sample_rate as f64;

            for mut w in result.words {
                w.start += offset_sec;
                w.end += offset_sec;
                w.speaker = Some(speaker_id as u32);
                all_words.push(w);
            }

            if !full_text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&result.text);
        }

        Ok(TranscribeResult {
            text: full_text,
            words: all_words,
            duration_s: samples.len() as f64 / self.info.sample_rate as f64,
        })
    }

    #[cfg(feature = "diarization")]
    pub fn transcribe_samples_with_diarization(
        &self,
        samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, SiamError> {
        let mut result = self.transcribe_samples(samples, triplet)?;
        if let Some(ref diarization) = self.diarization {
            match diarization.diarize(samples) {
                Ok(turns) => {
                    crate::diarization::assign_speakers(&mut result.words, &turns);
                }
                Err(e) => {
                    tracing::warn!("Diarization failed: {e}");
                }
            }
        }
        Ok(result)
    }

    pub fn process_chunk(
        &self,
        chunk: &[f32],
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscriptSegment>, SiamError> {
        state.audio_buffer.extend_from_slice(chunk);

        if !state.should_process() {
            return Ok(vec![]);
        }

        let samples = std::mem::take(&mut state.audio_buffer);
        let result = match self.transcribe_samples(&samples, triplet) {
            Ok(r) => r,
            Err(e) => {
                state.audio_buffer = samples;
                return Err(e);
            }
        };

        let overlap_start = samples.len().saturating_sub(state.overlap_samples);
        state.audio_buffer = samples[overlap_start..].to_vec();

        let now = super::streaming::now_timestamp();
        Ok(vec![TranscriptSegment {
            text: Arc::new(result.text),
            words: Arc::new(result.words),
            is_final: false,
            timestamp: now,
        }])
    }

    pub fn create_state(&self, _diarization: bool) -> Result<StreamingState, SiamError> {
        Ok(StreamingState::new())
    }

    pub fn flush_state(
        &self,
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Option<TranscriptSegment> {
        if state.audio_buffer.is_empty() {
            return None;
        }
        let samples = std::mem::take(&mut state.audio_buffer);
        match self.transcribe_samples(&samples, triplet) {
            Ok(result) => {
                let now = super::streaming::now_timestamp();
                Some(TranscriptSegment {
                    text: Arc::new(result.text),
                    words: Arc::new(result.words),
                    is_final: true,
                    timestamp: now,
                })
            }
            Err(e) => {
                tracing::warn!("Flush failed, preserving audio buffer: {e}");
                state.audio_buffer = samples;
                None
            }
        }
    }
}
