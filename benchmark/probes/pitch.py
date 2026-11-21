"""Pitch-contour features per meow (Schötz et al., Meowsic: food meows tend to rise, vet meows to fall) for CatMeows and
the real-world clips. -> data/wild/pitch_features.npz"""

import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "benchmark")
import librosa  # noqa: E402
from wild import WILD, labelled, local_name  # noqa: E402

from meow_decoder import bench, core  # noqa: E402

NAMES = ["rise_st", "slope_st_per_s", "sd_st", "median_hz", "voiced_s", "peak_pos"]


def contour(w):
    f0, voiced, _ = librosa.pyin(w, fmin=200, fmax=2000, sr=core.SR, frame_length=1024, hop_length=160)
    f = f0[voiced & ~np.isnan(f0)]
    t = np.where(voiced & ~np.isnan(f0))[0] * 0.01
    return f, t


def features(w):
    rows = []
    for a, b in bench.segments(w)[:3]:  # the three loudest sound events
        f, t = contour(w[a:b])
        if len(f) < 8:
            continue
        st = 12 * np.log2(f / 440)
        third = max(len(st) // 3, 1)
        rows.append([st[-third:].mean() - st[:third].mean(), np.polyfit(t, st, 1)[0], st.std(), np.median(f), len(f) * 0.01,
                     float(np.argmax(st)) / len(st)])  # fmt: skip
    return np.mean(rows, 0) if rows else np.full(len(NAMES), np.nan)


def main():
    files, y, cats, _ = bench.catmeows("data/dataset")
    C = np.array([features(core.load(f)) for f in files])
    W, meta = [], []
    for r in labelled():
        W.append(features(core.decode((WILD / "audio" / local_name(r)).read_bytes())))
        meta.append({"label": r["label"], "uploader": r["artist"], "title": r["title"]})
    np.savez("data/wild/pitch_features.npz", catmeows=C, y=y, cats=cats, wild=np.array(W), meta=json.dumps(meta), names=NAMES)


if __name__ == "__main__":
    main()
