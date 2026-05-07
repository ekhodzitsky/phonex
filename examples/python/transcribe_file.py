#!/usr/bin/env python3
"""Example: transcribe a WAV file using phonex Python bindings."""

import sys
from pathlib import Path

# Add bindings to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bindings" / "python"))

from phonex import Engine

MODEL_DIR = "models/sherpa-onnx-zipformer-thai-2024-06-20"
WAV_PATH = "test.wav"


def main():
    print(f"Loading model from {MODEL_DIR}...")
    with Engine(MODEL_DIR) as engine:
        print(f"Transcribing {WAV_PATH}...")
        text = engine.transcribe(WAV_PATH)
        print(f"Result: {text}")


if __name__ == "__main__":
    main()
