#!/usr/bin/env python3
"""Smoke tests for the phonex C-ABI FFI layer.

Run after building the shared library:
    cargo build --features ffi
    python tests/ffi_smoke.py

Expects the library at:
    target/debug/libphonex.{so,dylib}   (or release/ if CARGO_PROFILE=release)
"""

import ctypes
import os
import platform
import sys
from pathlib import Path


def find_library() -> Path:
    """Locate libphonex shared library relative to project root."""
    profile = os.environ.get("CARGO_PROFILE", "debug")
    root = Path(__file__).parent.parent
    target_dir = root / "target" / profile

    system = platform.system()
    if system == "Darwin":
        name = "libphonex.dylib"
    elif system == "Linux":
        name = "libphonex.so"
    elif system == "Windows":
        name = "phonex.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    candidate = target_dir / name
    if not candidate.exists():
        raise FileNotFoundError(
            f"Shared library not found: {candidate}\n"
            f"Build it first: cargo build --features ffi"
        )
    return candidate


def load_lib() -> ctypes.CDLL:
    lib_path = find_library()
    # On macOS, help the dynamic linker find transitive deps (ort, etc.)
    if platform.system() == "Darwin":
        os.environ.setdefault("DYLD_LIBRARY_PATH", str(lib_path.parent))
    return ctypes.CDLL(str(lib_path))


def test_engine_new_null():
    lib = load_lib()
    lib.phonex_engine_new.restype = ctypes.c_void_p
    result = lib.phonex_engine_new(None)
    assert not result, f"expected NULL for null model_dir, got {result}"
    print("✓ phonex_engine_new(NULL) -> NULL")


def test_engine_new_nonexistent():
    lib = load_lib()
    lib.phonex_engine_new.restype = ctypes.c_void_p
    result = lib.phonex_engine_new(b"/nonexistent/path/models")
    assert not result, f"expected NULL for missing models, got {result}"
    print("✓ phonex_engine_new('/nonexistent') -> NULL")


def test_string_free_null():
    lib = load_lib()
    # Should be a no-op, not a crash.
    lib.phonex_string_free(None)
    print("✓ phonex_string_free(NULL) -> no crash")


def test_stream_new_null():
    lib = load_lib()
    lib.phonex_stream_new.restype = ctypes.c_void_p
    result = lib.phonex_stream_new(None, None)
    assert not result, f"expected NULL for null args, got {result}"
    print("✓ phonex_stream_new(null, null) -> NULL")


def test_stream_process_chunk_null():
    lib = load_lib()
    lib.phonex_stream_process_chunk.restype = ctypes.c_void_p
    result = lib.phonex_stream_process_chunk(
        None,  # stream
        None,  # samples
        0,     # len
    )
    assert not result, f"expected NULL for null args, got {result}"
    print("✓ phonex_stream_process_chunk(null...) -> NULL")


def test_stream_flush_null():
    lib = load_lib()
    lib.phonex_stream_flush.restype = ctypes.c_void_p
    result = lib.phonex_stream_flush(None)
    assert not result, f"expected NULL for null args, got {result}"
    print("✓ phonex_stream_flush(null) -> NULL")


def main():
    print("phonex FFI smoke tests")
    print("-" * 40)
    test_engine_new_null()
    test_engine_new_nonexistent()
    test_string_free_null()
    test_stream_new_null()
    test_stream_process_chunk_null()
    test_stream_flush_null()
    print("-" * 40)
    print("All FFI smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
