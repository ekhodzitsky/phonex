use ndarray::{Array1, Array2, Axis};
use realfft::RealFftPlanner;

/// Audio preprocessor: resample + compute log-mel spectrogram.
/// Matches NeMo's AudioToMelSpectrogramPreprocessor as closely as possible.
pub struct AudioPreprocessor {
    target_sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    _n_mels: usize,
    window: Array1<f32>,
    mel_filterbank: Array2<f32>,
    preemph: f32,
    log_zero_guard: f32,
}

impl AudioPreprocessor {
    pub fn new(
        target_sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        win_length: usize,
        n_mels: usize,
        mel_filterbank: Array2<f32>,
    ) -> Self {
        let window = hann_window(win_length);
        Self {
            target_sample_rate,
            n_fft,
            hop_length,
            win_length,
            _n_mels: n_mels,
            window,
            mel_filterbank,
            preemph: 0.97,
            log_zero_guard: 2f32.powi(-24),
        }
    }

    /// Default parameters matching Typhoon ASR preprocessor.
    pub fn typhoon() -> Self {
        let fb = Self::load_mel_filterbank("models/mel_filterbank.bin", 80, 257);
        Self::new(16000, 512, 160, 400, 80, fb)
    }

    fn load_mel_filterbank(path: &str, n_mels: usize, n_freqs: usize) -> Array2<f32> {
        let bytes = std::fs::read(path).expect("Failed to read mel filterbank");
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Array2::from_shape_vec((n_mels, n_freqs), floats).expect("Invalid mel filterbank shape")
    }

    /// Simple linear interpolation resampler.
    pub fn resample(&self, samples: &[f32], from_rate: usize) -> crate::Result<Vec<f32>> {
        if from_rate == self.target_sample_rate {
            return Ok(samples.to_vec());
        }
        let ratio = from_rate as f32 / self.target_sample_rate as f32;
        if !(1.0 / 16.0..=16.0).contains(&ratio) {
            return Err(crate::SiamError::Inference(format!(
                "Resample ratio {ratio} out of safe bounds (1/16 .. 16)"
            )));
        }
        let out_len = (samples.len() as f32 / ratio) as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f32 * ratio;
            let src_idx_floor = src_idx.floor() as usize;
            let frac = src_idx - src_idx.floor();
            let s0 = samples.get(src_idx_floor).copied().unwrap_or(0.0);
            let s1 = samples.get(src_idx_floor + 1).copied().unwrap_or(s0);
            out.push(frac.mul_add(s1 - s0, s0));
        }
        Ok(out)
    }

    /// Compute log-mel spectrogram from audio samples at target sample rate.
    pub fn compute_mel(&self, samples: &[f32]) -> Array2<f32> {
        let preemphasized = self.preemphasis(samples);
        let stft = self.stft(&preemphasized);
        let power = stft.mapv(|x| x * x); // magnitude squared
        let mel = self.mel_filterbank.dot(&power);

        mel.mapv(|x| (x + self.log_zero_guard).ln())
    }

    fn preemphasis(&self, samples: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples.len());
        if samples.is_empty() {
            return out;
        }
        out.push(samples[0]);
        for i in 1..samples.len() {
            out.push((-self.preemph).mul_add(samples[i - 1], samples[i]));
        }
        out
    }

    fn stft(&self, samples: &[f32]) -> Array2<f32> {
        // Center padding like torch.stft(center=True)
        let pad_size = self.n_fft / 2;
        let mut padded = vec![0.0f32; pad_size];
        padded.extend_from_slice(samples);
        padded.resize(pad_size + samples.len() + pad_size, 0.0);

        let n_frames = (padded.len() - self.n_fft) / self.hop_length + 1;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let mut spectrum = fft.make_output_vec();
        let mut stft = Array2::zeros((self.n_fft / 2 + 1, n_frames));

        for i in 0..n_frames {
            let start = i * self.hop_length;
            let end = start + self.win_length;
            if end > padded.len() {
                break;
            }
            let mut frame: Vec<f32> = padded[start..end]
                .iter()
                .zip(self.window.iter())
                .map(|(s, w)| s * w)
                .collect();
            frame.resize(self.n_fft, 0.0);
            fft.process(&mut frame, &mut spectrum).unwrap();
            for (j, c) in spectrum.iter().enumerate() {
                stft[[j, i]] = c.norm();
            }
        }
        stft
    }

    /// Normalize per feature (per mel bin) across time.
    pub fn normalize_per_feature(&self, mel: &mut Array2<f32>) {
        for mut row in mel.axis_iter_mut(Axis(0)) {
            let mean = row.mean().unwrap_or(0.0);
            let std = row.std(0.0).max(1e-5);
            row.mapv_inplace(|x| (x - mean) / std);
        }
    }

    /// Read a WAV file and return (samples, sample_rate).
    pub fn read_wav(path: &str) -> crate::Result<(Vec<f32>, usize)> {
        let mut reader = hound::WavReader::open(path).map_err(|e| {
            crate::SiamError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("WAV read error: {e}"),
            ))
        })?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate as usize;
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
            }
            hound::SampleFormat::Int => {
                let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap_or(0) as f32 / max_val)
                    .collect()
            }
        };
        // Convert to mono if stereo
        let channels = spec.channels as usize;
        let mono: Vec<f32> = if channels > 1 {
            samples
                .chunks(channels)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            samples
        };
        Ok((mono, sample_rate))
    }
}

fn hann_window(size: usize) -> Array1<f32> {
    Array1::from_iter((0..size).map(|i| {
        (-0.5f32).mul_add(
            (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos(),
            0.5,
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let w = hann_window(401);
        assert_eq!(w.len(), 401);
        assert!((w[0] - 0.0).abs() < 1e-6);
        assert!((w[200] - 1.0).abs() < 1e-6);
    }
}
