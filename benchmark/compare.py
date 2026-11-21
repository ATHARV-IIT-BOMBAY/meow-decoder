"""Head-to-head on the same clips: MeowSense (github.com/sbor3937/MeowSense), the open cat-meow classifier closest to this
project, run with its own code and its best published setup (frozen AST embeddings + SVM-RBF, and its LogReg probe),
trained on all of CatMeows as it is, then scored on the real-world test (benchmark/wild.py) and on our unseen-cat folds.

    git clone https://github.com/sbor3937/MeowSense data/meowsense   # then check out meowsense_commit from compare_report.json
    python benchmark/compare.py        # -> benchmark/compare_report.json

Meow-Omni 1 (smgjch/Meow-Omni-1) isn't compared: its "intents" are the cat's next physical action from collar
sensors (lying, walking, grooming, litter box...), not why it meows, and it needs a ~20 GB GPU. See README.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEOWSENSE = ROOT / "data" / "meowsense"
sys.path[:0] = [str(MEOWSENSE / "src"), str(ROOT), str(ROOT / "benchmark")]

from embeddings import BACKBONES, _embed_batch, _load_backbone  # noqa: E402
from features import load_wav, scan_dataset  # noqa: E402
from train_transfer import build_probes  # noqa: E402
from wild import WILD, labelled, local_name  # noqa: E402

from meow_decoder import bench  # noqa: E402


def main():
    import subprocess

    commit = subprocess.run(["git", "-C", str(MEOWSENSE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    cfg = BACKBONES["ast"]
    processor, model, torch = _load_backbone(cfg)

    def embed(paths):
        return np.concatenate([_embed_batch(processor, model, torch, cfg, [load_wav(p, target_sr=16000)[0] for p in paths[i : i + 8]])
                               for i in range(0, len(paths), 8)])  # fmt: skip

    recs, rows = scan_dataset(ROOT / "data" / "dataset"), labelled()
    Xc, Xw = embed([r.path for r in recs]), embed([WILD / "audio" / local_name(r) for r in rows])
    yc, cats = np.array([r.context for r in recs]), np.array([r.cat_id for r in recs])
    ours = {o["title"]: o for o in json.loads((ROOT / "benchmark" / "wild_report.json").read_text())["clips_detail"]}
    lab = np.array([r["label"] for r in rows])
    meow = np.array([ours[r["title"]]["has_meow"] for r in rows])
    s = meow & np.isin(lab, list("FBI"))  # exactly the meows Meow Decoder is scored on

    def score(pred, sel):
        return {"top1": float((pred[sel] == lab[sel]).mean()),
                "by_label": {k: f"{int((pred[sel & (lab == k)] == k).sum())}/{int((sel & (lab == k)).sum())}" for k in "FBI"},
                "macro_recall": float(np.mean([(pred[sel & (lab == k)] == k).mean() for k in "FBI"]))}  # fmt: skip

    report = {"meowsense_commit": commit, "real_world_meows": int(s.sum()), "models": {}}
    ours_pred = np.array([ours[r["title"]]["results"][0][0] for r in rows])
    report["models"]["Meow Decoder v0.5 (untaught: pitch contour)"] = {"real_world": score(ours_pred, s)}
    folds = bench.folds_for(cats)
    for name, probe in build_probes(42).items():
        P = np.empty(len(yc), object)
        for f in np.unique(folds):
            probe.fit(Xc[folds != f], yc[folds != f])
            P[folds == f] = probe.predict(Xc[folds == f])
        probe.fit(Xc, yc)
        pred = probe.predict(Xw)
        report["models"][f"MeowSense AST + {name}"] = {
            "catmeows_unseen_cats": float((P == yc).mean()),
            "real_world": score(pred, s),
            "called_real_meows": {k: int((pred[meow] == k).sum()) for k in "FBI"},
        }
    Path(__file__).with_name("compare_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
