use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::TensorRef;

/// Wrapper around Silero VAD ONNX model.
///
/// The model expects 512-sample chunks at 16kHz. Internally it uses a 64-sample
/// context buffer and an LSTM state (2×1×128) that must be carried across calls.
pub struct SileroVad {
    session: Session,
    state: Array3<f32>,
    context: Array2<f32>,
    sample_rate: i64,
    /// Pre-allocated input buffer [1, 576] (64 context + 512 samples).
    input_buf: Array2<f32>,
}

impl SileroVad {
    pub fn new(model_path: &str) -> crate::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self {
            session,
            state: Array3::zeros((2, 1, 128)),
            context: Array2::zeros((1, 64)),
            sample_rate: 16000,
            input_buf: Array2::zeros((1, 576)),
        })
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
    }

    /// Process a single 512-sample window.
    ///
    /// Returns the speech probability in [0.0, 1.0].
    pub fn process(&mut self, samples: &[f32]) -> crate::Result<f32> {
        assert_eq!(samples.len(), 512, "SileroVAD expects exactly 512 samples");

        // Fill pre-allocated input buffer: context (64) + new samples (512)
        self.input_buf
            .slice_mut(ndarray::s![.., ..64])
            .assign(&self.context);
        for (i, &s) in samples.iter().enumerate() {
            self.input_buf[[0, 64 + i]] = s;
        }

        let input_tensor = TensorRef::from_array_view(self.input_buf.view())?;
        let state_tensor = TensorRef::from_array_view(self.state.view())?;
        let sr_array = ndarray::Array0::from_elem((), self.sample_rate);
        let sr_tensor = TensorRef::from_array_view(sr_array.view())?;

        let outputs = self.session.run(ort::inputs![
            "input" => input_tensor,
            "state" => state_tensor,
            "sr" => sr_tensor,
        ])?;

        let (_, prob_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let prob = prob_data.first().copied().unwrap_or(0.0);

        let (_, state_data) = outputs["stateN"].try_extract_tensor::<f32>()?;
        let state_out = ndarray::Array3::from_shape_vec((2, 1, 128), state_data.to_vec())?;
        self.state.assign(&state_out);

        // Update context to the last 64 samples of the current input window.
        self.context
            .assign(&self.input_buf.slice(ndarray::s![.., 512..]));

        Ok(prob)
    }
}

/// Simple VAD wrapper for legacy engine API.
pub struct Vad {
    segmenter: VadSegmenter,
}

impl Vad {
    pub fn new(config: VadConfig) -> crate::Result<Self> {
        let segmenter = VadSegmenter::new(&config.model_path)
            .map_err(|e| crate::SiamError::Inference(format!("Failed to load VAD model: {e}")))?
            .with_thresholds(
                config.speech_threshold,
                config.min_speech_duration_ms,
                config.min_silence_duration_ms,
                config.speech_pad_ms,
            );
        Ok(Self { segmenter })
    }

    pub fn split(&mut self, samples: &[f32], _sample_rate: usize) -> Vec<(usize, usize)> {
        self.segmenter.segment(samples).unwrap_or_default()
    }
}

/// Configuration for VAD.
pub struct VadConfig {
    pub model_path: String,
    pub speech_threshold: f32,
    pub min_speech_duration_ms: u32,
    pub min_silence_duration_ms: u32,
    pub speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            model_path: "models/silero_vad.onnx".into(),
            speech_threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 300,
            speech_pad_ms: 250,
        }
    }
}

/// Segment audio into speech / non-speech regions using Silero VAD.
pub struct VadSegmenter {
    vad: SileroVad,
    speech_threshold: f32,
    min_speech_duration_ms: u32,
    min_silence_duration_ms: u32,
    speech_pad_ms: u32,
    sample_rate: u32,
}

impl VadSegmenter {
    pub fn new(vad_model_path: &str) -> crate::Result<Self> {
        Ok(Self {
            vad: SileroVad::new(vad_model_path)?,
            speech_threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 300,
            speech_pad_ms: 250,
            sample_rate: 16000,
        })
    }

    /// Set thresholds (optional).
    pub fn with_thresholds(
        mut self,
        speech_threshold: f32,
        min_speech_ms: u32,
        min_silence_ms: u32,
        speech_pad_ms: u32,
    ) -> Self {
        self.speech_threshold = speech_threshold;
        self.min_speech_duration_ms = min_speech_ms;
        self.min_silence_duration_ms = min_silence_ms;
        self.speech_pad_ms = speech_pad_ms;
        self
    }

    /// Analyze `samples` (16 kHz, mono) and return speech segment boundaries
    /// as `(start_sample, end_sample)` pairs.
    pub fn segment(&mut self, samples: &[f32]) -> crate::Result<Vec<(usize, usize)>> {
        if samples.iter().any(|s| !s.is_finite()) {
            return Err(crate::SiamError::Inference(
                "VAD input contains NaN or infinite samples".into(),
            ));
        }
        self.vad.reset();

        let window_samples = 512usize;
        let num_windows = samples.len() / window_samples;
        let mut probs = Vec::with_capacity(num_windows);

        for i in 0..num_windows {
            let chunk = &samples[i * window_samples..(i + 1) * window_samples];
            let prob = self.vad.process(chunk)?;
            probs.push(prob);
        }

        // Convert thresholds from milliseconds to window counts.
        let ms_per_window = (window_samples as f32 / self.sample_rate as f32) * 1000.0;
        let min_speech_windows =
            (self.min_speech_duration_ms as f32 / ms_per_window).ceil() as usize;
        let min_silence_windows =
            (self.min_silence_duration_ms as f32 / ms_per_window).ceil() as usize;
        let pad_samples =
            (self.speech_pad_ms as f32 / 1000.0 * self.sample_rate as f32).ceil() as usize;

        // Hysteresis-based segmentation.
        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut seg_start = 0usize;
        let mut silence_count = 0usize;

        for (i, &prob) in probs.iter().enumerate() {
            if in_speech {
                if prob < self.speech_threshold {
                    silence_count += 1;
                    if silence_count >= min_silence_windows {
                        let seg_end = (i + 1) * window_samples;
                        let duration_windows = i + 1 - seg_start / window_samples;
                        if duration_windows >= min_speech_windows {
                            segments.push((seg_start, seg_end));
                        }
                        in_speech = false;
                        silence_count = 0;
                    }
                } else {
                    silence_count = 0;
                }
            } else {
                if prob >= self.speech_threshold {
                    seg_start = i * window_samples;
                    in_speech = true;
                    silence_count = 0;
                }
            }
        }

        // Close trailing segment.
        if in_speech {
            let seg_end = num_windows * window_samples;
            let duration_windows = num_windows - seg_start / window_samples;
            if duration_windows >= min_speech_windows {
                segments.push((seg_start, seg_end));
            }
        }

        // Apply padding and clamp to audio bounds.
        let audio_len = samples.len();
        let padded: Vec<(usize, usize)> = segments
            .into_iter()
            .map(|(s, e)| {
                (
                    s.saturating_sub(pad_samples),
                    (e + pad_samples).min(audio_len),
                )
            })
            .collect();

        Ok(padded)
    }
}

/// Streaming VAD state machine for real-time pipelines.
///
/// Buffers incoming audio into 512-sample windows (32 ms @ 16 kHz) and runs
/// hysteresis to suppress rapid toggling between silence and speech.
pub struct StreamingVad {
    vad: SileroVad,
    buffer: Vec<f32>,
    state: VadState,
    speech_threshold: f32,
    silence_threshold: f32,
    min_speech_frames: usize,
    min_silence_frames: usize,
    speech_frame_count: usize,
    silence_frame_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum VadState {
    Silence,
    Speech,
}

impl StreamingVad {
    pub fn new(vad_model_path: &str) -> crate::Result<Self> {
        Ok(Self {
            vad: SileroVad::new(vad_model_path)?,
            buffer: Vec::new(),
            state: VadState::Silence,
            speech_threshold: 0.5,
            silence_threshold: 0.35,
            min_speech_frames: 3,
            min_silence_frames: 10,
            speech_frame_count: 0,
            silence_frame_count: 0,
        })
    }

    /// Process raw audio samples and return the subset classified as speech.
    ///
    /// Samples are buffered internally until a full 512-sample window is
    /// available.  Returned samples correspond to windows in the `Speech`
    /// state (including the grace-period windows before silence is
    /// confirmed).  An empty vector means no speech was detected in this
    /// batch.
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        self.buffer.extend_from_slice(samples);
        let mut output = Vec::new();
        const WINDOW_SIZE: usize = 512;

        while self.buffer.len() >= WINDOW_SIZE {
            let prob = self.vad.process(&self.buffer[..WINDOW_SIZE]).unwrap_or(0.0);

            match self.state {
                VadState::Silence => {
                    if prob > self.speech_threshold {
                        self.speech_frame_count += 1;
                        if self.speech_frame_count >= self.min_speech_frames {
                            self.state = VadState::Speech;
                            self.speech_frame_count = 0;
                            self.silence_frame_count = 0;
                            output.extend_from_slice(&self.buffer[..WINDOW_SIZE]);
                        }
                    } else {
                        self.speech_frame_count = 0;
                    }
                }
                VadState::Speech => {
                    if prob < self.silence_threshold {
                        self.silence_frame_count += 1;
                        if self.silence_frame_count >= self.min_silence_frames {
                            self.state = VadState::Silence;
                            self.silence_frame_count = 0;
                            self.speech_frame_count = 0;
                            // Do not include the silence window that triggered
                            // the transition.
                        } else {
                            output.extend_from_slice(&self.buffer[..WINDOW_SIZE]);
                        }
                    } else {
                        self.silence_frame_count = 0;
                        output.extend_from_slice(&self.buffer[..WINDOW_SIZE]);
                    }
                }
            }

            self.buffer.drain(..WINDOW_SIZE);
        }

        output
    }

    /// Process samples and detect speech→silence transitions.
    ///
    /// Returns `(speech_samples, speech_ended)` where `speech_ended` is `true`
    /// if the VAD transitioned from Speech to Silence during this batch.
    pub fn process_with_transitions(&mut self, samples: &[f32]) -> (Vec<f32>, bool) {
        let was_speech = self.state == VadState::Speech;
        let output = self.process(samples);
        let speech_ended = was_speech && self.state == VadState::Silence;
        (output, speech_ended)
    }

    /// Reset the VAD state machine and the underlying Silero model.
    pub fn reset(&mut self) {
        self.vad.reset();
        self.buffer.clear();
        self.state = VadState::Silence;
        self.speech_frame_count = 0;
        self.silence_frame_count = 0;
    }

    /// Returns `true` when the state machine is currently in `Speech`.
    pub fn is_speech(&self) -> bool {
        matches!(self.state, VadState::Speech)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioPreprocessor;

    const VAD_MODEL: &str = "models/silero_vad.onnx";

    macro_rules! skip_if_no_vad_model {
        () => {
            if !std::path::Path::new(VAD_MODEL).exists() {
                eprintln!("Skipping test: VAD model not found");
                return;
            }
        };
    }

    #[test]
    fn test_vad_config_default() {
        let config = VadConfig::default();
        assert_eq!(config.model_path, "models/silero_vad.onnx");
        assert_eq!(config.speech_threshold, 0.5);
        assert_eq!(config.min_speech_duration_ms, 250);
        assert_eq!(config.min_silence_duration_ms, 300);
        assert_eq!(config.speech_pad_ms, 250);
    }

    #[test]
    fn test_vad_segmenter_rejects_nan() {
        skip_if_no_vad_model!();
        let mut segmenter = VadSegmenter::new(VAD_MODEL).unwrap();
        let mut samples = vec![0.0f32; 512];
        samples[10] = f32::NAN;
        let result = segmenter.segment(&samples);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN") || err.contains("infinite"));
    }

    #[test]
    fn test_vad_segmenter_rejects_inf() {
        skip_if_no_vad_model!();
        let mut segmenter = VadSegmenter::new(VAD_MODEL).unwrap();
        let mut samples = vec![0.0f32; 512];
        samples[10] = f32::INFINITY;
        let result = segmenter.segment(&samples);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN") || err.contains("infinite"));
    }

    #[test]
    fn test_streaming_vad_state_machine() {
        skip_if_no_vad_model!();
        let mut vad = StreamingVad::new(VAD_MODEL).unwrap();

        // All zeros should keep the state machine in Silence.
        let zeros = vec![0.0f32; 512 * 5];
        let out = vad.process(&zeros);
        assert!(out.is_empty());
        assert_eq!(vad.state, VadState::Silence);

        // Real speech audio should trigger a transition to Speech.
        let (samples, _sr) = AudioPreprocessor::read_wav(
            "models/sherpa-onnx-zipformer-thai-2024-06-20/test_wavs/0.wav",
        )
        .unwrap();
        let out = vad.process(&samples);
        assert!(!out.is_empty());
        assert_eq!(vad.state, VadState::Speech);
    }
}
