#!/usr/bin/env python3
"""Generate mel filterbank compatible with Sherpa-ONNX / icefall feature extraction."""

import torch
import torchaudio
import numpy as np


def main():
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=257,
        f_min=20.0,
        f_max=7600.0,
        n_mels=80,
        sample_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    # Transpose to [n_mels, n_freqs] for Rust compatibility
    mel_fb_t = mel_fb.t().numpy().astype(np.float32)

    out_path = "models/mel_filterbank.bin"
    with open(out_path, "wb") as f:
        f.write(mel_fb_t.tobytes())

    loaded = np.fromfile(out_path, dtype=np.float32).reshape(80, 257)
    assert np.allclose(mel_fb_t, loaded)
    print(f"Saved {out_path} ({mel_fb_t.shape})")


if __name__ == "__main__":
    main()
