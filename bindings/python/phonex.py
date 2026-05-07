"""Python bindings for phonex C-ABI FFI layer."""

import ctypes
import json
import os
import platform
from pathlib import Path
from typing import List, Optional


class PhonexError(Exception):
    """ phonex error. """
    pass


def _find_library() -> Path:
    """Locate libphonex shared library relative to project root."""
    system = platform.system()
    if system == "Darwin":
        name = "libphonex.dylib"
    elif system == "Linux":
        name = "libphonex.so"
    elif system == "Windows":
        name = "phonex.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    # Try common locations
    candidates = [
        Path(__file__).parent.parent.parent / "target" / "release" / name,
        Path(__file__).parent.parent.parent / "target" / "debug" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Shared library not found. Build it first: "
        f"cargo build --release --features ffi --no-default-features"
    )


class _Lib:
    def __init__(self):
        lib_path = _find_library()
        if platform.system() == "Darwin":
            os.environ.setdefault("DYLD_LIBRARY_PATH", str(lib_path.parent))
        self._lib = ctypes.CDLL(str(lib_path))

        # Engine
        self._lib.phonex_engine_new.argtypes = [ctypes.c_char_p]
        self._lib.phonex_engine_new.restype = ctypes.c_void_p

        self._lib.phonex_engine_new_with_pool_size.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self._lib.phonex_engine_new_with_pool_size.restype = ctypes.c_void_p

        self._lib.phonex_engine_free.argtypes = [ctypes.c_void_p]
        self._lib.phonex_engine_free.restype = None

        # Transcribe
        self._lib.phonex_transcribe_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.phonex_transcribe_file.restype = ctypes.c_char_p

        # Streaming
        self._lib.phonex_stream_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.phonex_stream_new.restype = ctypes.c_void_p

        self._lib.phonex_stream_process_chunk.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
        ]
        self._lib.phonex_stream_process_chunk.restype = ctypes.c_char_p

        self._lib.phonex_stream_flush.argtypes = [ctypes.c_void_p]
        self._lib.phonex_stream_flush.restype = ctypes.c_char_p

        self._lib.phonex_stream_flush_with_tokens.argtypes = [ctypes.c_void_p]
        self._lib.phonex_stream_flush_with_tokens.restype = ctypes.c_char_p

        self._lib.phonex_stream_reset.argtypes = [ctypes.c_void_p]
        self._lib.phonex_stream_reset.restype = None

        self._lib.phonex_stream_free.argtypes = [ctypes.c_void_p]
        self._lib.phonex_stream_free.restype = None

        # Utility
        self._lib.phonex_string_free.argtypes = [ctypes.c_char_p]
        self._lib.phonex_string_free.restype = None

    def engine_new(self, model_dir: str, pool_size: int = 1) -> int:
        ptr = self._lib.phonex_engine_new_with_pool_size(
            model_dir.encode("utf-8"), pool_size
        )
        if not ptr:
            raise PhonexError(f"Failed to load engine from {model_dir}")
        return ptr

    def engine_free(self, engine: int) -> None:
        self._lib.phonex_engine_free(engine)

    def transcribe_file(self, engine: int, wav_path: str) -> str:
        c_text = self._lib.phonex_transcribe_file(
            engine, wav_path.encode("utf-8")
        )
        if not c_text:
            raise PhonexError(f"Transcription failed for {wav_path}")
        text = ctypes.cast(c_text, ctypes.c_char_p).value.decode("utf-8")
        self._lib.phonex_string_free(c_text)
        return text

    def stream_new(self, model_dir: str, vad_path: Optional[str] = None) -> int:
        vad = vad_path.encode("utf-8") if vad_path else None
        ptr = self._lib.phonex_stream_new(model_dir.encode("utf-8"), vad)
        if not ptr:
            raise PhonexError(f"Failed to create stream from {model_dir}")
        return ptr

    def stream_process_chunk(self, stream: int, samples: List[float]) -> List[dict]:
        arr = (ctypes.c_float * len(samples))(*samples)
        c_json = self._lib.phonex_stream_process_chunk(stream, arr, len(samples))
        if not c_json:
            raise PhonexError("Stream processing failed")
        json_str = ctypes.cast(c_json, ctypes.c_char_p).value.decode("utf-8")
        self._lib.phonex_string_free(c_json)
        return json.loads(json_str)

    def stream_flush(self, stream: int) -> str:
        c_text = self._lib.phonex_stream_flush(stream)
        if not c_text:
            raise PhonexError("Stream flush failed")
        text = ctypes.cast(c_text, ctypes.c_char_p).value.decode("utf-8")
        self._lib.phonex_string_free(c_text)
        return text

    def stream_flush_with_tokens(self, stream: int) -> dict:
        c_json = self._lib.phonex_stream_flush_with_tokens(stream)
        if not c_json:
            raise PhonexError("Stream flush with tokens failed")
        json_str = ctypes.cast(c_json, ctypes.c_char_p).value.decode("utf-8")
        self._lib.phonex_string_free(c_json)
        return json.loads(json_str)

    def stream_reset(self, stream: int) -> None:
        self._lib.phonex_stream_reset(stream)

    def stream_free(self, stream: int) -> None:
        self._lib.phonex_stream_free(stream)


_lib = None


def _get_lib() -> _Lib:
    global _lib
    if _lib is None:
        _lib = _Lib()
    return _lib


class Engine:
    """Offline inference engine."""

    def __init__(self, model_dir: str, pool_size: int = 1):
        self._lib = _get_lib()
        self._ptr = self._lib.engine_new(model_dir, pool_size)

    def transcribe(self, wav_path: str) -> str:
        return self._lib.transcribe_file(self._ptr, wav_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._ptr:
            self._lib.engine_free(self._ptr)
            self._ptr = 0


class Stream:
    """Real-time streaming transcription session."""

    def __init__(self, model_dir: str, vad_path: Optional[str] = None):
        self._lib = _get_lib()
        self._ptr = self._lib.stream_new(model_dir, vad_path)

    def process_chunk(self, samples: List[float]) -> List[dict]:
        return self._lib.stream_process_chunk(self._ptr, samples)

    def flush(self) -> str:
        return self._lib.stream_flush(self._ptr)

    def flush_with_tokens(self) -> dict:
        return self._lib.stream_flush_with_tokens(self._ptr)

    def reset(self) -> None:
        self._lib.stream_reset(self._ptr)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._ptr:
            self._lib.stream_free(self._ptr)
            self._ptr = 0
