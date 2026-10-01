#!/usr/bin/env python3
"""Minimal Proto-Germanic TTS.

    python pgmc_tts.py "ˈhun.dɑz" out.wav
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from normalize import normalize, to_ids

MODEL = Path(__file__).parent / "pgmc.onnx"
HEAD_MS, TAIL_MS = 250, 400          # pad: level-based tools trim edges
LENGTH_SCALE = 0.85                  # the speaker read deliberately; 1.0 is slow


def say(ipa, model=MODEL, length_scale=LENGTH_SCALE):
    cfg = json.loads(Path(str(model) + ".json").read_text(encoding="utf-8"))
    ids, unknown = to_ids(normalize(ipa), cfg["phoneme_id_map"])
    if unknown:
        print(f"warning: no id for {unknown} - dropped", file=sys.stderr)
    inf = cfg["inference"]
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    audio = sess.run(None, {
        "input": np.array([ids], dtype=np.int64),
        "input_lengths": np.array([len(ids)], dtype=np.int64),
        "scales": np.array([inf["noise_scale"],
                            inf["length_scale"] * length_scale,
                            inf["noise_w"]], dtype=np.float32),
    })[0].squeeze()
    rate = cfg["audio"]["sample_rate"]
    audio = audio / max(float(np.max(np.abs(audio))), 1e-9) * 0.9
    return np.concatenate([
        np.zeros(int(rate * HEAD_MS / 1000), dtype=np.float32),
        audio.astype(np.float32),
        np.zeros(int(rate * TAIL_MS / 1000), dtype=np.float32)]), rate


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, sr = say(sys.argv[1])
    sf.write(sys.argv[2], a, sr, subtype="PCM_16")
    print(f"{sys.argv[2]}  {len(a)/sr:.2f}s @ {sr} Hz")
