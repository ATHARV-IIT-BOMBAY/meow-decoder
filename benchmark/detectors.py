"""Purring, hissing and yowling detectors: small logistic regressions on the BEATs embedding, replacing fixed
AudioSet thresholds, which missed most real hisses and fired on loud meows.

Training audio: FSD50K's cat clips (CC licences, labels checked by people), a sample of FSD50K's other sounds
(non-cat hisses, speech, everything else), fight and mating recordings from Freesound (benchmark/detector_labels.csv)
and CatMeows. Nothing from an uploader in the real-world test (benchmark/wild_labels.csv), which stays a clean test.

    python benchmark/detectors.py features   # embed everything once (cached in data/detectors/)
    python benchmark/detectors.py build      # -> meow_decoder/detectors.npz, benchmark/detector_report.json
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))
from wild import UA, WILD, labelled, local_name  # noqa: E402

from meow_decoder import bench, core  # noqa: E402

DATA, FSD = Path("data/detectors"), Path("data/fsd50k")
EXTRA = Path(__file__).with_name("detector_labels.csv")
TARGET_FALSE_ALARMS = 0.01  # on training negatives, out of fold
# Shipped: hissing and yowling, which beat AudioSet's scores on the real-world test. Purring stays with AudioSet: the detector
# caught 29 of 30 real purrs (AudioSet 26) but with 15 false alarms instead of 9, so it isn't clearly better.
SHIP = {"Hiss", "Caterwaul"}
# detector -> (positive labels, labels left out of training and scoring for it: they contain the sound or imitate it)
TASKS = {
    "Purr": ({"purr"}, {"human"}),
    "Hiss": ({"hiss"}, {"human", "growl", "yowl"}),
    "Caterwaul": ({"yowl", "heat"}, {"human", "hiss", "growl"}),
}


def fsd_clips(test_ids, test_uploaders, n_other=300, n_hiss=150):
    """FSD50K clips: every cat clip, all-but-test; plus a fixed random sample of non-cat hisses and of everything else."""
    info = {**json.load(open(FSD / "dev_clips_info_FSD50K.json")), **json.load(open(FSD / "eval_clips_info_FSD50K.json"))}
    rows = [(r, "dev") for r in csv.DictReader(open(FSD / "dev.csv"))] + [(r, "eval") for r in csv.DictReader(open(FSD / "eval.csv"))]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    out, n = [], {"hiss": 0, "other": 0}
    for r, split in rows:
        L, up = set(r["labels"].split(",")), info[r["fname"]]["uploader"]
        if r["fname"] in test_ids or up in test_uploaders:
            continue
        if "Cat" in L:
            label = "hiss" if "Hiss" in L else "growl" if "Growling" in L else "purr" if "Purr" in L else "meow" if "Meow" in L else "cat"
        elif "Hiss" in L and n["hiss"] < n_hiss:
            label, n["hiss"] = "not_cat_hiss", n["hiss"] + 1
        elif n["other"] < n_other and not L & {"Animal", "Hiss"}:
            label, n["other"] = "not_cat", n["other"] + 1
        else:
            continue
        out.append({"id": f"fsd:{r['fname']}", "label": label, "uploader": up, "source": "fsd50k", "fsd": (r["fname"], split)})
    return out


def clip_list():
    wild = labelled()
    test_ids = {r["title"].split()[1].rstrip(":") for r in wild if r["title"].startswith("Freesound")}
    test_uploaders = {r["artist"] for r in wild}
    clips = [{"id": "wild:" + r["title"], "label": r["label"], "uploader": r["artist"], "source": "wild",
              "path": str(WILD / "audio" / local_name(r))} for r in wild]  # fmt: skip
    clips += fsd_clips(test_ids, test_uploaders)
    fsd_ids = {c["fsd"][0]: c for c in clips if "fsd" in c}
    cand = {str(r["title"].split()[1].rstrip(":")): r for r in json.load(open(WILD / "freesound_candidates.json"))}
    for r in csv.DictReader(open(EXTRA)):
        assert r["uploader"] not in test_uploaders, r
        if r["freesound_id"] in fsd_ids:  # already in FSD50K: relabel it
            fsd_ids[r["freesound_id"]]["label"] = r["label"]
            continue
        clips.append({"id": "extra:" + r["freesound_id"], "label": r["label"], "uploader": r["uploader"], "source": "freesound",
                      "url": cand[r["freesound_id"]]["url"], "path": str(DATA / "audio" / f"{r['freesound_id']}.wav")})  # fmt: skip
    files, y, cats, _ = bench.catmeows("data/dataset")
    clips += [{"id": "catmeows:" + f.name, "label": "catmeows", "uploader": "catmeows:" + c, "source": "catmeows", "path": str(f)}
              for f, c in zip(files, cats)]  # fmt: skip
    return clips


def ast_all(wave):
    import torch

    fe, ast, dev, _ = core._ast()
    windows = [wave[i : i + 10 * core.SR] for i in range(0, max(len(wave) - core.SR // 2, 1), 10 * core.SR)]
    with torch.no_grad():
        logits = torch.cat([ast(**fe(w, sampling_rate=core.SR, return_tensors="pt").to(dev)).logits for w in windows])
    return torch.sigmoid(logits).max(0).values.cpu().numpy()


def features(_):
    from huggingface_hub import hf_hub_download

    DATA.mkdir(parents=True, exist_ok=True)
    cache_file = DATA / "features.npz"
    cache = dict(np.load(cache_file, allow_pickle=False)) if cache_file.exists() else {}
    clips = clip_list()
    for i, c in enumerate(clips):
        if f"beats|{c['id']}" in cache:
            continue
        if "fsd" in c:
            path = Path(hf_hub_download("Fhrozen/FSD50k", f"clips/{c['fsd'][1]}/{c['fsd'][0]}.wav", repo_type="dataset"))
        else:
            path = Path(c["path"])
            if c["source"] == "freesound" and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-user_agent", UA, "-i", c["url"], "-t", "30", "-vn", "-ac", "1",
                                "-ar", "16000", str(path)], check=True, timeout=120)  # fmt: skip
        wave = core.load(path) if c["source"] == "catmeows" else core.decode(path.read_bytes())
        cache[f"beats|{c['id']}"], cache[f"ast|{c['id']}"] = core.embedding(wave, "beats"), ast_all(wave)
        if i % 25 == 0:
            np.savez(cache_file, **cache)
            print(i, len(clips), flush=True)
    np.savez(cache_file, **cache)
    (DATA / "clips.json").write_text(json.dumps(clips, indent=0))
    print(len(clips), "clips embedded")


def fit(X, yb):
    from sklearn.linear_model import LogisticRegression

    mean, scale = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit((X - mean) / scale, yb)
    return mean, scale, m.coef_[0], m.intercept_[0]


def score(model, X):
    mean, scale, coef, intercept = model
    return ((X - mean) / scale) @ coef + intercept


def threshold(X, yb, groups):
    """Score at which TARGET_FALSE_ALARMS of the training negatives fire, from out-of-fold scores."""
    from sklearn.model_selection import GroupKFold

    oof = np.zeros(len(yb))
    for tr, te in GroupKFold(n_splits=5).split(X, yb, groups):
        oof[te] = score(fit(X[tr], yb[tr]), X[te])
    return float(np.quantile(oof[~yb], 1 - TARGET_FALSE_ALARMS))


def build(_):
    from sklearn.model_selection import GroupKFold
    from transformers import AutoConfig

    clips = json.loads((DATA / "clips.json").read_text())
    cache = np.load(DATA / "features.npz")
    B = np.array([cache[f"beats|{c['id']}"] for c in clips])
    A = np.array([cache[f"ast|{c['id']}"] for c in clips])
    lab, src, up = (np.array([c[k] for c in clips]) for k in ("label", "source", "uploader"))
    col = {v: int(k) for k, v in AutoConfig.from_pretrained(core.AST_ID, revision=core.AST_REVISION).id2label.items()}
    train, test = src != "wild", src == "wild"
    saved, report = {}, {"training_clips": {s: int((src == s).sum()) for s in np.unique(src[train])}, "detectors": {}}
    for key, (pos, ignore) in TASKS.items():
        yb, use = np.isin(lab, list(pos)), ~np.isin(lab, list(ignore))
        t = train & use
        model, th = fit(B[t], yb[t]), threshold(B[t], yb[t], up[t])
        if key in SHIP:
            saved.update({f"{key}_{n}": v for n, v in zip(("mean", "scale", "coef", "intercept"), model)} | {f"{key}_threshold": th})
        # held-out checks inside the training sources: each uploader (or CatMeows cat) scored by a model that never heard it
        oof = np.zeros(len(yb))
        idx = np.where(t)[0]
        for tr, te in GroupKFold(n_splits=10).split(idx, yb[idx], up[idx]):
            m_ = fit(B[idx[tr]], yb[idx[tr]])
            th_ = threshold(B[idx[tr]], yb[idx[tr]], up[idx[tr]])
            oof[idx[te]] = score(m_, B[idx[te]]) - th_
        fired = score(model, B) > th
        rule = A[:, col[key]] >= core.SIGNALS[key][3]
        w = test & use
        rate = lambda m, sel: f"{int(m[sel].sum())}/{int(sel.sum())}"  # noqa: E731
        report["detectors"][key] = {
            "shipped": key in SHIP,
            "threshold": th,
            "real_world_test": {"caught": rate(fired, w & yb), "false_alarms": rate(fired, w & ~yb),
                                "audioset_rule_caught": rate(rule, w & yb), "audioset_rule_false_alarms": rate(rule, w & ~yb)},
            "held_out_uploaders": {s: {"caught": rate(oof > 0, t & yb & (src == s)), "false_alarms": rate(oof > 0, t & ~yb & (src == s)),
                                       "audioset_rule_caught": rate(rule, t & yb & (src == s)),
                                       "audioset_rule_false_alarms": rate(rule, t & ~yb & (src == s))}
                                   for s in ("fsd50k", "freesound", "catmeows")},
            "non_cat_sounds_false_alarms": rate(oof > 0, t & np.isin(lab, ["not_cat", "not_cat_hiss"])),
            "non_cat_audioset_rule_false_alarms": rate(rule, t & np.isin(lab, ["not_cat", "not_cat_hiss"])),
        }  # fmt: skip
        print(key, json.dumps(report["detectors"][key]))
    np.savez_compressed(core.DETECTORS, **saved)
    Path(__file__).with_name("detector_report.json").write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    sub.add_parser("features").set_defaults(fn=features)
    sub.add_parser("build").set_defaults(fn=build)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
