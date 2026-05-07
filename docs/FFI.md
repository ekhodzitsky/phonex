# FFI Reference

phonex exposes a C-ABI FFI layer for embedding into Android, iOS, Python, Node.js (via NAPI), or any C-compatible runtime.

## Building

### macOS / Linux

```bash
cargo build --release --features ffi --no-default-features
```

Output:
- `target/release/libphonex.dylib` (macOS)
- `target/release/libphonex.so` (Linux)
- `target/release/phonex.dll` (Windows)

### Android (arm64)

Requires [cargo-ndk](https://github.com/bbqsrc/cargo-ndk):

```bash
cargo ndk -t arm64-v8a -o ./jniLibs build --release \
  --no-default-features --features ffi
```

## API Reference

### Engine

#### `phonex_engine_new`

```c
void* phonex_engine_new(const char* model_dir);
```

Load models from `model_dir` with default pool size (1).

- **Returns**: opaque engine handle, or `NULL` on error.

#### `phonex_engine_new_with_pool_size`

```c
void* phonex_engine_new_with_pool_size(const char* model_dir, size_t pool_size);
```

Load models with custom session pool size.

- `pool_size = 1`: ~350 MB RAM (recommended for mobile)
- `pool_size = 4`: ~560 MB RAM (default desktop)

#### `phonex_transcribe_file`

```c
char* phonex_transcribe_file(void* engine, const char* wav_path);
```

Transcribe a WAV file synchronously.

- **Returns**: newly allocated UTF-8 string, or `NULL` on error.
- **Ownership**: caller must free with `phonex_string_free`.

#### `phonex_engine_free`

```c
void phonex_engine_free(void* engine);
```

Release the engine and all ONNX sessions. No-op if `engine` is `NULL`.

### Streaming

#### `phonex_stream_new`

```c
void* phonex_stream_new(const char* model_dir, const char* vad_path);
```

Create a streaming pipeline.

- `model_dir`: path to a Sherpa-ONNX streaming Zipformer model.
- `vad_path`: path to `silero_vad.onnx`, or `NULL` for no VAD.

#### `phonex_stream_process_chunk`

```c
char* phonex_stream_process_chunk(void* stream, const float* samples, size_t len);
```

Feed f32 audio samples (16 kHz mono) and return newly emitted tokens as JSON.

- **Returns**: JSON array of tokens, or `NULL` on error.
- **Ownership**: caller must free with `phonex_string_free`.

#### `phonex_stream_flush`

```c
char* phonex_stream_flush(void* stream);
```

Finalize and return the full transcript.

- **Returns**: UTF-8 string, or `NULL` on error.
- **Ownership**: caller must free with `phonex_string_free`.

#### `phonex_stream_flush_with_tokens`

```c
char* phonex_stream_flush_with_tokens(void* stream);
```

Finalize and return the full transcript **with word-level timestamps**.

- **Returns**: JSON string `{"text":"...","tokens":[{"id":1,"text":"hello","start":0.0,"end":0.5,"confidence":0.98}]}`
- **Ownership**: caller must free with `phonex_string_free`.

#### `phonex_stream_reset`

```c
void phonex_stream_reset(void* stream);
```

Reset the pipeline for a new utterance. No-op if `stream` is `NULL`.

#### `phonex_stream_free`

```c
void phonex_stream_free(void* stream);
```

Release the streaming pipeline. No-op if `stream` is `NULL`.

### Utility

#### `phonex_string_free`

```c
void phonex_string_free(char* s);
```

Free any string returned by phonex functions. No-op if `s` is `NULL`.

## Python Example

```python
import ctypes

lib = ctypes.CDLL("target/release/libphonex.dylib")

lib.phonex_engine_new.restype = ctypes.c_void_p
lib.phonex_transcribe_file.restype = ctypes.c_char_p
lib.phonex_string_free.argtypes = [ctypes.c_char_p]

engine = lib.phonex_engine_new(b"models/sherpa-onnx-zipformer-thai-2024-06-20")
text = lib.phonex_transcribe_file(engine, b"audio.wav")
print(text.decode("utf-8"))
lib.phonex_string_free(text)
lib.phonex_engine_free(engine)
```

## Kotlin / Android Example

```kotlin
class PhonexLib {
    companion object {
        init {
            System.loadLibrary("phonex")
        }
    }

    external fun phonex_engine_new(modelDir: String): Long
    external fun phonex_transcribe_file(engine: Long, wavPath: String): String?
    external fun phonex_string_free(s: String?)
    external fun phonex_engine_free(engine: Long)
}
```

## Thread Safety

- `phonex_engine_new*` and `phonex_engine_free` are **not** thread-safe. Create one engine per thread or protect with a mutex.
- `phonex_transcribe_file` is thread-safe if the engine was created with `pool_size > 1`.
- Each `phonex_stream_*` instance is **not** thread-safe. Use one stream per thread.
