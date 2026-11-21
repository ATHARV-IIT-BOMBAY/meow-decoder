"""Build the pitch-contour situation model shipped in meow_decoder/pitch.npz, and score it on unseen cats.

Used for cats nobody has taught yet: on real home recordings the embedding model's guess was right 14% of the time
(benchmark/wild.py), because it learned CatMeows' recording setup. Pitch contour shape (Meowsic: food meows tend to
rise, vet meows to fall) carries over. See README, "Tested on real recordings from the internet".

    python benchmark/pitch.py --data data/dataset     # -> meow_decoder/pitch.npz, benchmark/pitch_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meow_decoder import bench, core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    a = ap.parse_args()
    files, y, cats, _ = bench.catmeows(a.data)
    F = np.array([core.pitch_features(core.load(f)) for f in files])
    folds, P = bench.folds_for(cats), np.zeros((len(y), 3))
    for f in np.unique(folds):
        h = core.fit_pitch_head(F[folds != f], y[folds != f])
        P[folds == f] = h.proba_features(F[folds == f])
    report = {"unseen_cats": bench.metrics(P, y, cats), "no_pitch_found": int(np.isnan(F).any(1).sum()), "clips": len(y)}
    head = core.fit_pitch_head(F, y)
    head.cats = cats
    head.save(core.PITCH_MODEL)
    Path(__file__).with_name("pitch_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report["unseen_cats"].items() if k != "recall"}, indent=1, default=float))


if __name__ == "__main__":
    main()
