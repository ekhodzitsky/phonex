//! Audio resampling and file decoding.

use crate::error::SiamError;

/// High-quality polyphase FIR resampler (rubato 2.0 Async sinc).
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, SiamError> {
    if samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return Ok(Vec::new());
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let samples: Vec<f32> = samples.iter().map(|&s| if s.is_finite() { s } else { 0.0 }).collect();

    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{
        Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
        WindowFunction, calculate_cutoff,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let channels = 1;
    const CHUNK_SAMPLES: usize = 16000 * 5;
    let chunk_size = samples.len().min(CHUNK_SAMPLES);

    let sinc_len = 128;
    let oversampling_factor = 256;
    let interpolation = SincInterpolationType::Linear;
    let window = WindowFunction::BlackmanHarris2;
    let f_cutoff = calculate_cutoff(sinc_len, window);
    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation,
        oversampling_factor,
        window,
    };

    let mut resampler = Async::<f32>::new_sinc(ratio, 1.1, &params, chunk_size, channels, FixedAsync::Input)
        .map_err(|e| SiamError::Audio(format!("Resampler init failed: {e}")))?;

    let output_capacity = (samples.len() as f64 * ratio) as usize + samples.len();
    let mut outdata = vec![0.0f32; output_capacity * channels];

    let input_adapter = InterleavedSlice::new(&samples, channels, samples.len())
        .map_err(|e| SiamError::Audio(format!("Resampler input adapter failed: {e}")))?;
    let outdata_capacity = outdata.len() / channels;
    let mut output_adapter = InterleavedSlice::new_mut(&mut outdata, channels, outdata_capacity)
        .map_err(|e| SiamError::Audio(format!("Resampler output adapter failed: {e}")))?;

    let mut indexing = Indexing {
        input_offset: 0,
        output_offset: 0,
        active_channels_mask: None,
        partial_len: None,
    };

    let mut input_frames_left = samples.len();
    let mut input_frames_next = resampler.input_frames_next();

    while input_frames_left >= input_frames_next {
        let (nbr_in, nbr_out) = resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
            .map_err(|e| SiamError::Audio(format!("Resampling failed: {e}")))?;
        indexing.input_offset += nbr_in;
        indexing.output_offset += nbr_out;
        input_frames_left -= nbr_in;
        input_frames_next = resampler.input_frames_next();
    }

    indexing.partial_len = Some(input_frames_left);
    let (_nbr_in, _nbr_out) = resampler
        .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
        .map_err(|e| SiamError::Audio(format!("Resampling final chunk failed: {e}")))?;

    let delay = resampler.output_delay();
    let expected_out = (samples.len() as f64 * ratio) as usize;
    Ok(outdata.into_iter().skip(delay * channels).take(expected_out * channels).collect())
}

/// Decode an audio file (wav, mp3, ogg, flac, aac, etc.) to mono f32 samples at any sample rate.
/// Returns `(samples, sample_rate)`.
pub fn decode_audio(data: &[u8]) -> Result<(Vec<f32>, u32), SiamError> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(data.to_vec())), Default::default());
    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| SiamError::Audio(format!("Failed to probe audio format: {e}")))?;

    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| SiamError::Audio("No audio track found".into()))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| SiamError::Audio(format!("Failed to create decoder: {e}")))?;

    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(pkt) => pkt,
            Err(SymphoniaError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymphoniaError::ResetRequired) => continue,
            Err(e) => return Err(SiamError::Audio(format!("Read error: {e}"))),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                if sample_buf.is_none() {
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }
                if let Some(ref mut buf) = sample_buf {
                    buf.copy_interleaved_ref(decoded);
                    let samples = buf.samples();
                    if channels > 1 {
                        // Convert interleaved stereo to mono
                        all_samples.extend(
                            samples.chunks(channels).map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                        );
                    } else {
                        all_samples.extend_from_slice(samples);
                    }
                }
            }
            Err(SymphoniaError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => continue,
            Err(e) => return Err(SiamError::Audio(format!("Decode error: {e}"))),
        }
    }

    Ok((all_samples, sample_rate))
}
