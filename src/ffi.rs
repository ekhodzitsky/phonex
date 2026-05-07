//! C-ABI FFI layer for embedding phonex into other languages.
//!
//! Exposes a minimal surface so that Python, Kotlin, Swift, or C can:
//! 1. Load the inference engine (`phonex_engine_new`).
//! 2. Transcribe an audio file (`phonex_transcribe_file`).
//! 3. Stream audio in real-time (`phonex_stream_new`, `phonex_stream_process_chunk`,
//!    `phonex_stream_flush`).
//! 4. Free the returned C string (`phonex_string_free`).
//! 5. Tear down the engine / stream (`phonex_engine_free`, `phonex_stream_free`).
//!
//! All functions are `unsafe` by nature (raw pointers cross the FFI boundary) but
//! the implementation checks nulls and logs errors before returning sentinel values.

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::inference::Engine;
use crate::model_config::ModelInfo;
use crate::streaming_pipeline::StreamingPipeline;

/// Opaque handle to the inference engine.
pub struct PhonexEngine {
    engine: Engine,
    disposed: AtomicBool,
}

/// Opaque handle to a streaming transcription session.
pub struct PhonexStream {
    pipeline: StreamingPipeline,
    disposed: AtomicBool,
}

/// Load the ONNX models from `model_dir` and create an inference engine.
///
/// Uses the default pool size (1). For desktop/server use, prefer
/// `phonex_engine_new_with_pool_size` with a larger pool.
///
/// # Safety
/// `model_dir` must be a valid, null-terminated UTF-8 string.
/// Returns a pointer to a `PhonexEngine` on success, or `NULL` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_engine_new(model_dir: *const c_char) -> *mut PhonexEngine {
    unsafe { phonex_engine_new_with_pool_size(model_dir, 1) }
}

/// Load the ONNX models with a custom session pool size.
///
/// `pool_size` controls how many concurrent inference sessions are kept in
/// memory. Each session loads the full encoder/decoder/joiner, so RAM scales
/// linearly.
///
/// # Safety
/// `model_dir` must be a valid, null-terminated UTF-8 string.
/// Returns a pointer to a `PhonexEngine` on success, or `NULL` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_engine_new_with_pool_size(
    model_dir: *const c_char,
    pool_size: usize,
) -> *mut PhonexEngine {
    if model_dir.is_null() {
        eprintln!("phonex_engine_new_with_pool_size: model_dir is null");
        return ptr::null_mut();
    }

    let dir_str = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("phonex_engine_new_with_pool_size: model_dir is not valid UTF-8: {e}");
            return ptr::null_mut();
        }
    };

    match Engine::load_with_pool_size(dir_str, pool_size) {
        Ok(engine) => {
            let handle = Box::new(PhonexEngine {
                engine,
                disposed: AtomicBool::new(false),
            });
            Box::into_raw(handle)
        }
        Err(e) => {
            eprintln!("phonex_engine_new_with_pool_size: failed to load engine: {e}");
            ptr::null_mut()
        }
    }
}

/// Transcribe an audio file and return the recognized text as a newly allocated C string.
///
/// # Safety
/// - `engine` must be a non-null pointer returned by `phonex_engine_new` and not yet freed.
/// - `wav_path` must be a valid, null-terminated UTF-8 string.
///
/// Returns a pointer to a NUL-terminated UTF-8 string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `phonex_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_transcribe_file(
    engine: *mut PhonexEngine,
    wav_path: *const c_char,
) -> *mut c_char {
    if engine.is_null() {
        eprintln!("phonex_transcribe_file: engine is null");
        return ptr::null_mut();
    }
    if wav_path.is_null() {
        eprintln!("phonex_transcribe_file: wav_path is null");
        return ptr::null_mut();
    }

    let path_str = match unsafe { CStr::from_ptr(wav_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("phonex_transcribe_file: wav_path is not valid UTF-8: {e}");
            return ptr::null_mut();
        }
    };

    let engine_ref = unsafe { &(*engine).engine };

    match engine_ref.transcribe_file(path_str) {
        Ok(text) => match CString::new(text) {
            Ok(cstr) => cstr.into_raw(),
            Err(e) => {
                eprintln!("phonex_transcribe_file: result contains interior NUL: {e}");
                ptr::null_mut()
            }
        },
        Err(e) => {
            eprintln!("phonex_transcribe_file: transcription failed: {e}");
            ptr::null_mut()
        }
    }
}

/// Free a C string previously returned by `phonex_transcribe_file` or the
/// streaming functions.
///
/// # Safety
/// `s` must be a pointer returned by one of the transcription functions and not
/// yet freed, or `NULL` (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_string_free(s: *mut c_char) {
    if !s.is_null() {
        let _ = unsafe { CString::from_raw(s) };
    }
}

/// Free an inference engine previously created by `phonex_engine_new`.
///
/// # Safety
/// `engine` must be a pointer returned by `phonex_engine_new` and not yet freed,
/// or `NULL` (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_engine_free(engine: *mut PhonexEngine) {
    if !engine.is_null() {
        let disposed = unsafe { std::ptr::addr_of_mut!((*engine).disposed) };
        if unsafe { (*disposed).swap(true, Ordering::Relaxed) } {
            return;
        }
        let _ = unsafe { Box::from_raw(engine) };
    }
}

// ---------------------------------------------------------------------------
// Streaming API
// ---------------------------------------------------------------------------

/// Create a new streaming transcription session.
///
/// # Safety
/// `model_dir` must be a valid, null-terminated UTF-8 string.
/// `vad_path` may be NULL (no VAD) or a path to `silero_vad.onnx`.
/// Returns a pointer to a `PhonexStream` on success, or `NULL` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_stream_new(
    model_dir: *const c_char,
    vad_path: *const c_char,
) -> *mut PhonexStream {
    if model_dir.is_null() {
        eprintln!("phonex_stream_new: model_dir is null");
        return ptr::null_mut();
    }

    let dir_str = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("phonex_stream_new: model_dir is not valid UTF-8: {e}");
            return ptr::null_mut();
        }
    };

    let vad_str = if vad_path.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(vad_path) }.to_str() {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("phonex_stream_new: vad_path is not valid UTF-8: {e}");
                return ptr::null_mut();
            }
        }
    };

    let info = match ModelInfo::from_model_dir(dir_str) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("phonex_stream_new: failed to load model info: {e}");
            return ptr::null_mut();
        }
    };

    match StreamingPipeline::from_model_dir(dir_str, &info, vad_str) {
        Ok(pipeline) => {
            let handle = Box::new(PhonexStream {
                pipeline,
                disposed: AtomicBool::new(false),
            });
            Box::into_raw(handle)
        }
        Err(e) => {
            eprintln!("phonex_stream_new: failed to create streaming pipeline: {e}");
            ptr::null_mut()
        }
    }
}

/// Process a chunk of f32 audio samples and return any newly emitted tokens.
///
/// # Safety
/// - `stream` must be a valid pointer.
/// - `samples` must point to at least `len` valid f32 values (16 kHz mono).
///
/// Returns a newly allocated JSON array string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `phonex_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_stream_process_chunk(
    stream: *mut PhonexStream,
    samples: *const f32,
    len: usize,
) -> *mut c_char {
    if stream.is_null() {
        eprintln!("phonex_stream_process_chunk: stream is null");
        return ptr::null_mut();
    }
    if samples.is_null() {
        eprintln!("phonex_stream_process_chunk: samples is null");
        return ptr::null_mut();
    }

    let stream_ref = unsafe { &mut (*stream).pipeline };
    let sample_slice = unsafe { std::slice::from_raw_parts(samples, len) };

    match stream_ref.accept_audio(sample_slice) {
        Ok(tokens) => {
            let json = serde_json::to_string(&tokens).unwrap_or_else(|_| "[]".into());
            match CString::new(json) {
                Ok(cstr) => cstr.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        }
        Err(e) => {
            eprintln!("phonex_stream_process_chunk: inference failed: {e}");
            ptr::null_mut()
        }
    }
}

/// Flush the streaming pipeline and return the final text.
///
/// # Safety
/// `stream` must be a valid pointer.
///
/// Returns a newly allocated C string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `phonex_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_stream_flush(stream: *mut PhonexStream) -> *mut c_char {
    if stream.is_null() {
        eprintln!("phonex_stream_flush: stream is null");
        return ptr::null_mut();
    }

    let stream_ref = unsafe { &mut (*stream).pipeline };

    match stream_ref.flush() {
        Ok(text) => match CString::new(text) {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        Err(e) => {
            eprintln!("phonex_stream_flush: flush failed: {e}");
            ptr::null_mut()
        }
    }
}

/// Flush the streaming pipeline and return the final text with word-level timestamps.
///
/// # Safety
/// `stream` must be a valid pointer.
///
/// Returns a newly allocated JSON string on success, or `NULL` on failure.
/// Format: `{"text":"hello world","tokens":[{"id":1,"text":"hello","start":0.0,"end":0.5,"confidence":0.98},...]}`
/// The caller **must** free the returned string with `phonex_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_stream_flush_with_tokens(stream: *mut PhonexStream) -> *mut c_char {
    if stream.is_null() {
        eprintln!("phonex_stream_flush_with_tokens: stream is null");
        return ptr::null_mut();
    }

    let stream_ref = unsafe { &mut (*stream).pipeline };

    match stream_ref.flush_with_tokens() {
        Ok((text, tokens)) => {
            let result = serde_json::json!({
                "text": text,
                "tokens": tokens,
            });
            let json = serde_json::to_string(&result).unwrap_or_else(|_| "{}".into());
            match CString::new(json) {
                Ok(cstr) => cstr.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        }
        Err(e) => {
            eprintln!("phonex_stream_flush_with_tokens: flush failed: {e}");
            ptr::null_mut()
        }
    }
}

/// Reset the streaming pipeline for a new utterance.
///
/// # Safety
/// `stream` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_stream_reset(stream: *mut PhonexStream) {
    if stream.is_null() {
        return;
    }
    let stream_ref = unsafe { &mut (*stream).pipeline };
    stream_ref.reset();
}

/// Free a streaming session.
///
/// # Safety
/// `stream` must be a pointer returned by `phonex_stream_new` and not yet freed,
/// or `NULL` (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phonex_stream_free(stream: *mut PhonexStream) {
    if !stream.is_null() {
        let disposed = unsafe { std::ptr::addr_of_mut!((*stream).disposed) };
        if unsafe { (*disposed).swap(true, Ordering::Relaxed) } {
            return;
        }
        let _ = unsafe { Box::from_raw(stream) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_new_null() {
        let engine = unsafe { phonex_engine_new(ptr::null()) };
        assert!(engine.is_null());
    }

    #[test]
    fn test_stream_new_null() {
        let stream = unsafe { phonex_stream_new(ptr::null(), ptr::null()) };
        assert!(stream.is_null());
    }

    #[test]
    fn test_string_free_null() {
        unsafe { phonex_string_free(ptr::null_mut()) };
    }

    #[test]
    fn test_engine_free_null() {
        unsafe { phonex_engine_free(ptr::null_mut()) };
    }

    #[test]
    fn test_stream_free_null() {
        unsafe { phonex_stream_free(ptr::null_mut()) };
    }
}
