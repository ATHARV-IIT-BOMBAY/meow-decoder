"""Everything the browser version (web/, GitHub Pages) needs, as one JSON file -> web/model.json.

The browser runs AudioSet's AST (Xenova's quantized ONNX copy, pinned below) instead of BEATs, so:
- untaught situation guesses come from the pitch model, as in the app;
- hissing and purring use AudioSet's scores and yowling the AST-mode detector (meow_decoder/detectors_ast.npz);
- teaching refits on CatMeows' AudioSet log-odds plus pitch features, the same way Head.personalized does.
  On CatMeows across sessions that taught 59% -> 65% (3 per situation), against 65% -> 72% with BEATs.

    python benchmark/export_web.py      # needs data/dataset and data/detectors/features.npz (benchmark/detectors.py)
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meow_decoder import bench, core  # noqa: E402

AST_ONNX = {"id": "Xenova/ast-finetuned-audioset-10-10-0.4593", "revision": "249a1fbf0286b40e7f1ed687a8ae396997bf7dc6", "dtype": "q8"}


def int16_columns(X):
    """Compact float matrix: per-column offset and step, values as base64 little-endian int16."""
    lo, hi = X.min(0), X.max(0)
    step = np.where(hi > lo, (hi - lo) / 65000, 1.0)
    q = np.round((X - lo) / step - 32500).astype("<i2")
    return {"rows": X.shape[0], "cols": X.shape[1], "lo": lo.round(6).tolist(), "step": step.tolist(),
            "data": base64.b64encode(q.tobytes()).decode()}  # fmt: skip


def main():
    from transformers import AutoConfig

    labels = AutoConfig.from_pretrained(core.AST_ID, revision=core.AST_REVISION).id2label
    col = {v: int(k) for k, v in labels.items()}
    files, y, cats, _ = bench.catmeows(ROOT / "data" / "dataset")
    cache = np.load(ROOT / "data" / "detectors" / "features.npz")
    A = np.clip(np.array([cache[f"ast|catmeows:{f.name}"] for f in files]), 1e-6, 1 - 1e-6)
    P = np.array([core.pitch_features(core.load(f)) for f in files])
    pitch = core.Head.load(core.PITCH_MODEL)
    P = np.where(np.isnan(P), pitch.mean, P)  # as proba_features does
    X = np.hstack([np.log(A / (1 - A)), P])
    yowl = dict(np.load(core.DETECTORS["ast"]))
    out = {
        "ast": AST_ONNX,
        "audioset_index": {k: col[k] for k in core.CAT_LABELS + [core.SPEECH]},
        "cat_labels": core.CAT_LABELS, "speech": core.SPEECH, "cat_gate": core.CAT_GATE, "meow_min": core.MEOW_MIN,
        "signals": {k: {"emoji": e, "title": t, "meaning": m, "threshold": th} for k, (e, t, m, th) in core.SIGNALS.items()},
        "describe": {k: list(v) for k, v in {**core.CONTEXTS, **core.PRESETS}.items()},
        "contexts": list(core.CONTEXTS), "presets": list(core.PRESETS),
        "pitch_head": {"classes": pitch.classes.tolist(), "mean": pitch.mean.tolist(), "scale": pitch.scale.tolist(),
                       "coef": pitch.coef.tolist(), "intercept": pitch.intercept.tolist()},
        "detectors": {k: {"mean": yowl[f"{k}_mean"].tolist(), "scale": yowl[f"{k}_scale"].tolist(), "coef": yowl[f"{k}_coef"].tolist(),
                          "intercept": float(yowl[f"{k}_intercept"]), "threshold": float(yowl[f"{k}_threshold"])}
                      for k in core.SIGNALS if f"{k}_coef" in yowl},
        "teach": {"C": core.Head.C, "weight": 20, "y": y.tolist(), "X": int16_columns(X)},
    }  # fmt: skip
    path = ROOT / "web" / "model.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{path}: {path.stat().st_size / 1e6:.2f} MB, teach X {X.shape}, detectors {list(out['detectors'])}")


if __name__ == "__main__":
    main()
