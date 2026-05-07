#!/usr/bin/env python3
"""Example: real-time streaming transcription using phonex Python bindings."""

import sys
import wave
from pathlib import Path

# Add bindings to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bindings" / "python"))

from phonex import Stream

MODEL_DIR = "models/sherpa-onnx-streaming-zipformer-en-2023-06-21"
WAV_PATH = "test.wav"
CHUNK_SIZE = 8000  # 500ms at 16kHz


def read_wav_chunks(path: str, chunk_size: int):
    with wave.open(path, "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()

        if framerate != 16000:
            raise ValueError(f"Expected 16kHz, got {framerate}Hz")
        if nchannels != 1:
            raise ValueError(f"Expected mono, got {nchannels} channels")
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit, got {sampwidth * 8}-bit")

        for offset in range(0, nframes, chunk_size):
            frames = wf.readframes(min(chunk_size, nframes - offset))
            samples = [
                int.from_bytes(frames[i:i + 2], "little", signed=True) / 32768.0
                for i in range(0, len(frames), 2)
            ]
            yield samples


def main():
    print(f"Loading streaming model from {MODEL_DIR}...")
    with Stream(MODEL_DIR) as stream:
        print(f"Streaming {WAV_PATH}...")
        for i, chunk in enumerate(read_wav_chunks(WAV_PATH, CHUNK_SIZE)):
            tokens = stream.process_chunk(chunk)
            if tokens:
                texts = [t["text"] for t in tokens]
                print(f"  [{i}] {' '.join(texts)}")

        print("Flushing...")
        text = stream.flush()
        print(f"Final: {text}")


if __name__ == "__main__":
    main()
