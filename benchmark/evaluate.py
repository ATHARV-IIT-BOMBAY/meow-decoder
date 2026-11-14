"""Full held-out accuracy report -> benchmark/report.json.

    python benchmark/evaluate.py [--data data/dataset] [--catsound data/catsound/CAT_SOUND_DB_SAMPLES]

Everything is measured on cats (or source recordings) the model never trained on.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from meow_decoder.bench import (
    augment_copy,
    benchmark,
    catmeows,
    embed_dataset,
    folds_for,
    metrics,
    robustness,
    segments,
    teaching_curve,
    unseen_cat_proba,
)
from meow_decoder.core import CONTEXTS, SR, Head, embedding, load


def mfcc_stats(w):
    import librosa

    m = librosa.feature.mfcc(y=w, sr=SR, n_mfcc=40)
    return np.r_[m.mean(1), m.std(1)]


def background(w):
    """Spectral colour of the quietest 20% of frames: what the room sounds like, with as little meow as possible."""
    import librosa

    rms = librosa.feature.rms(y=w, frame_length=512, hop_length=256)[0]
    m = librosa.feature.mfcc(y=w, sr=SR, n_mfcc=20, n_fft=512, hop_length=256)
    return m[:, rms <= np.quantile(rms, 0.2)].mean(1)


def fit_temperature(Z, y, classes):
    """Temperature scaling (tested, not shipped). Z are out-of-fold log-probabilities."""
    idx = np.searchsorted(classes, y)
    sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    return float(min(np.linspace(0.5, 4, 71), key=lambda T: -np.log(sm(Z / T)[np.arange(len(y)), idx]).mean()))


def calibration_table(P, y):
    classes = np.unique(y)
    conf, hit = P.max(1), classes[P.argmax(1)] == y
    bins = [(0.33, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    return [
        dict(lo=lo, hi=min(hi, 1), n=int(m.sum()), said=float(conf[m].mean()), right=float(hit[m].mean()))
        for lo, hi in bins
        if (m := (conf >= lo) & (conf < hi)).sum()
    ]


def paper_retest(root, encoder):
    """The Multi-QuadEmoNet protocol (augment, then split at random) vs keeping each recording's copies together."""
    files = sorted(f for d in Path(root).iterdir() if d.is_dir() for f in d.iterdir())
    labels = np.array([f.parent.name for f in files])
    rng = np.random.default_rng(0)
    X, Y, G = [], [], []
    for g, (f, lab) in enumerate(zip(files, labels)):
        w = load(f)
        for a in range(9):
            X.append(embedding(w if a == 0 else augment_copy(w, a % 3, rng), encoder))
            Y.append(lab)
            G.append(g)
    X, Y, G = np.stack(X), np.array(Y), np.array(G)
    order = rng.permutation(len(Y))
    random_folds, grouped_folds = np.empty(len(Y), int), G % 5
    random_folds[order] = np.arange(len(Y)) % 5

    def acc(folds):
        hit = []
        for f in range(5):
            h = Head.fit(X[folds != f], Y[folds != f])
            hit += list(h.classes[h.proba(X[folds == f]).argmax(1)] == Y[folds == f])
        return float(np.mean(hit))

    return {
        "clips": len(files),
        "classes": int(len(set(labels))),
        "copies_per_clip": 8,
        "augment_then_random_split": acc(random_folds),
        "copies_kept_together": acc(grouped_folds),
        "chance": 1 / len(set(labels)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--catsound", default="data/catsound/CAT_SOUND_DB_SAMPLES")
    ap.add_argument("--out", default="benchmark/report.json")
    a = ap.parse_args()

    files, y, cats, sessions = catmeows(a.data)
    classes = np.array(sorted(CONTEXTS))
    report = {
        "clips": len(y),
        "cats": int(len(set(cats))),
        "sessions_per_cat": {c: int(len(set(sessions[cats == c]))) for c in sorted(set(cats))},
    }

    # Headline, both encoders, trained exactly as `meow train` does
    emb = {}
    for enc in ["beats", "ast"]:
        X, yy, cc, extra = embed_dataset(files, y, cats, enc)
        emb[enc] = X[~extra]
        report[enc] = benchmark(X, yy, cc, extra)
    X = emb["beats"]
    P = unseen_cat_proba(X, y, cats)
    pred = classes[P.argmax(1)]
    report["confusion"] = [[int(((y == t) & (pred == p)).sum()) for p in classes] for t in classes]
    report["calibration"] = calibration_table(P, y)
    conf, hit = P.max(1), pred == y
    report["coverage"] = [
        dict(threshold=t, answered=float((conf >= t).mean()), accuracy=float(hit[conf >= t].mean())) for t in [0.5, 0.6, 0.7, 0.8]
    ]
    report["per_cat"] = sorted(([c, float(hit[cats == c].mean()), int((cats == c).sum())] for c in np.unique(cats)), key=lambda r: -r[1])
    report["seen_cats_top1"] = float(
        np.mean(
            [  # random 5-fold split: the same cats appear in training and test
                (
                    Head.fit(X[np.arange(len(y)) % 5 != f], y[np.arange(len(y)) % 5 != f]).proba(X[np.arange(len(y)) % 5 == f]).argmax(1)
                    == np.searchsorted(classes, y[np.arange(len(y)) % 5 == f])
                ).mean()
                for f in range(5)
            ]
        )
    )

    # Is it the cat or the room? (H2 in REVIEW.md)
    waves = [load(f) for f in files]

    def best_of(F):  # simple baselines get their best regularisation (an upper bound, conservative for our claims)
        old, out = Head.C, []
        for c in (0.001, 0.01, 0.1, 1.0):
            Head.C = c
            out.append(metrics(unseen_cat_proba(F, y, cats), y, cats, 0)["top1"])
        Head.C = old
        return max(out)

    report["baselines"] = {
        "majority": float(max(np.mean(y == c) for c in classes)),
        "mfcc": best_of(np.stack([mfcc_stats(w) for w in waves])),
        "background_only": best_of(np.stack([background(w) for w in waves])),
        "loudest_event_only": metrics(
            unseen_cat_proba(np.stack([embedding(w[slice(*segments(w)[0])], "beats") for w in waves]), y, cats), y, cats, 0
        )["top1"],
    }

    # Teaching a cat: across sessions (headline) and within one (optimistic), plus the weight choice on held-out cats
    report["teaching"] = {
        "other_session": teaching_curve(X, y, cats, sessions),
        "same_session": teaching_curve(X, y, cats, sessions, same_session=True),
    }
    u = np.unique(cats)
    halves = {"A": set(u[::2]), "B": set(u[1::2])}
    sweep = {
        h: {w: teaching_curve(X, y, cats, sessions, ks=(3,), weight=w, only=c)[0]["accuracy"] for w in [1, 5, 10, 20, 50]}
        for h, c in halves.items()
    }
    report["teaching"]["weight_sweep_k3"] = {h: {str(w): v for w, v in s.items()} for h, s in sweep.items()}

    # Temperature scaling, fitted on training cats only (tested, not shipped)
    folds = folds_for(cats)
    Pt = np.zeros_like(P)
    for f in range(5):
        tr = folds != f
        Z = np.log(np.clip(unseen_cat_proba(X[tr], y[tr], cats[tr]), 1e-12, 1))
        T = fit_temperature(Z, y[tr], classes)
        Pt[~tr] = np.exp(np.log(np.clip(P[~tr], 1e-12, 1)) / T)
        Pt[~tr] /= Pt[~tr].sum(1, keepdims=True)
    report["temperature_scaling"] = {"ece_before": metrics(P, y, cats, 0)["ece"], "ece_after": metrics(Pt, y, cats, 0)["ece"]}

    report["robustness"] = robustness(files, y, cats, "beats")
    if Path(a.catsound).is_dir():
        report["paper_retest"] = paper_retest(a.catsound, "beats")

    Path(a.out).write_text(json.dumps(report, indent=1))
    show = {k: report[k] for k in ("beats", "ast", "baselines", "teaching", "temperature_scaling", "paper_retest") if k in report}
    show["robustness"] = {k: round(v["top1"], 3) for k, v in report["robustness"].items()}
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    main()
