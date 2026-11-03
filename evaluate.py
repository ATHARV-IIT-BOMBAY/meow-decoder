"""Honest accuracy report for the meow model. Run: .venv/bin/python evaluate.py  -> report.json

Everything here is measured on cats (or source clips) the model never trained on.
"""
import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from meow import CONTEXTS, SR, analyze, load


def softmax(z):
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def calibrated(model, X, T):
    return softmax(model.decision_function(X) / T)


def fit_temperature(Z, y, classes):
    """Temperature scaling: T minimizing log-loss of out-of-fold scores (tested, not shipped: made overall calibration worse)."""
    idx = np.searchsorted(classes, y)
    return float(min(np.linspace(0.5, 4, 71), key=lambda T: -np.log(softmax(Z / T)[np.arange(len(y)), idx]).mean()))

rng = np.random.default_rng(0)
clf = lambda C=0.002: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
mfcc = lambda w: np.r_[librosa.feature.mfcc(y=w, sr=SR, n_mfcc=40).mean(1), librosa.feature.mfcc(y=w, sr=SR, n_mfcc=40).std(1)]

# ---------- CatMeows: 3 situations, 21 cats ----------
files = sorted(Path("data/dataset").glob("*.wav"))
waves = [load(f) for f in files]
y = np.array([f.name[0] for f in files])
cats = np.array([f.name.split("_")[1] for f in files])
X = np.stack([analyze(w)[0] for w in waves])
M = np.stack([mfcc(w) for w in waves])
classes = np.array(sorted(CONTEXTS))
unseen = GroupKFold(5)

P = cross_val_predict(clf(), X, y, cv=unseen, groups=cats, method="predict_proba")
# Calibrated: temperature fitted on each round's training cats only, then applied to its unseen test cats
Pc = np.zeros_like(P)
for tr, te in unseen.split(X, y, cats):
    Z = cross_val_predict(clf(), X[tr], y[tr], cv=GroupKFold(5), groups=cats[tr], method="decision_function")
    Pc[te] = calibrated(clf().fit(X[tr], y[tr]), X[te], fit_temperature(Z, y[tr], classes))
pred = classes[P.argmax(1)]
conf = P.max(1)
hit = pred == y
top2 = np.array([t in classes[np.argsort(p)[-2:]] for t, p in zip(y, P)])

pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
bins = [(0.33, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
def calibration(P):
    conf, hit = P.max(1), classes[P.argmax(1)] == y
    return [dict(lo=lo, hi=min(hi, 1), n=int(m.sum()), said=float(conf[m].mean()), right=float(hit[m].mean()))
            for lo, hi in bins if (m := (conf >= lo) & (conf < hi)).sum()]


calib, calib_after = calibration(P), calibration(Pc)
gap = lambda c: sum(b["n"] * abs(b["said"] - b["right"]) for b in c) / len(y)  # expected calibration error
coverage = [dict(threshold=t, answered=float((conf >= t).mean()), accuracy=float(hit[conf >= t].mean()))
            for t in [0.0, 0.5, 0.6, 0.7, 0.8] if (conf >= t).any()]
per_cat = sorted(([c, float(hit[cats == c].mean()), int((cats == c).sum())] for c in np.unique(cats)), key=lambda r: -r[1])


def acc(Xf, C=0.002):
    p = cross_val_predict(clf(C), Xf, y, cv=unseen, groups=cats)
    return float((p == y).mean())


baselines = {
    "Always guess the most common situation": float(np.mean(y == "I")),
    "Classic MFCC features (what the paper used)": max(acc(M, C) for C in [0.001, 0.01, 0.1, 1]),
    "Our model (bioacoustic BEATs embeddings)": float(hit.mean()),
}
seen_acc = float((cross_val_predict(clf(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)) == y).mean())

# personalization: k labelled meows per situation from the test cat, weight 20
curve = []
for k in [0, 1, 2, 3, 4]:
    hits = []
    for rep in range(5):
        for c in np.unique(cats):
            idx = np.where(cats == c)[0]
            teach = []
            for lab in np.unique(y[idx]):
                li = idx[y[idx] == lab]
                teach += list(rng.choice(li, min(k, len(li) - 1), replace=False))  # keep >=1 clip per situation for testing
            test = np.setdiff1d(idx, teach)
            tr = np.r_[np.where(cats != c)[0], teach].astype(int)
            w = np.r_[np.ones((cats != c).sum()), np.full(len(teach), 20)]
            m = clf().fit(X[tr], y[tr], logisticregression__sample_weight=w)
            hits += list(m.predict(X[test]) == y[test])
    curve.append(dict(k=k, accuracy=float(np.mean(hits))))

# ---------- Cat Sound V2 samples: 10 classes x 5 clips -> the paper's evaluation, reproduced ----------
root = Path("data/catsound/CAT_SOUND_DB_SAMPLES")
src_files = sorted(f for d in root.iterdir() if d.is_dir() for f in d.iterdir())
src_y = np.array([f.parent.name for f in src_files])


def augment(w):  # same three augmentations as the paper: noise, pitch shift, high-pass
    kind = rng.integers(3)
    if kind == 0:
        return w + rng.normal(0, 0.005, len(w)).astype(np.float32)
    if kind == 1:
        return librosa.effects.pitch_shift(w, sr=SR, n_steps=float(rng.uniform(-2, 2)))
    return librosa.effects.preemphasis(w, coef=0.97)


AUG = 8
feats, labels, groups = [], [], []
for g, (f, lab) in enumerate(zip(src_files, src_y)):
    w = load(f)
    for a in range(AUG + 1):
        feats.append(analyze(w if a == 0 else augment(w))[0])
        labels.append(lab)
        groups.append(g)
feats, labels, groups = np.stack(feats), np.array(labels), np.array(groups)
leaky = float((cross_val_predict(clf(0.01), feats, labels, cv=StratifiedKFold(5, shuffle=True, random_state=0)) == labels).mean())
honest = float((cross_val_predict(clf(0.01), feats, labels, cv=GroupKFold(5), groups=groups) == labels).mean())

report = dict(
    n_clips=len(y), n_cats=len(np.unique(cats)), classes={k: CONTEXTS[k][1] for k in classes}, class_counts={k: int((y == k).sum()) for k in classes},
    top1=float(hit.mean()), top2=float(top2.mean()), seen_cats_top1=seen_acc,
    per_class=[dict(key=k, precision=float(a), recall=float(b), f1=float(c), n=int(d)) for k, a, b, c, d in zip(classes, pr, rc, f1, sup)],
    confusion=confusion_matrix(y, pred, labels=classes).tolist(),
    calibration=calib, calibration_after=calib_after, ece=dict(before=gap(calib), after=gap(calib_after)), coverage=coverage, per_cat=per_cat, baselines=baselines, personalization=curve,
    paper_repro=dict(n_source=len(src_files), n_classes=len(np.unique(src_y)), aug_per_clip=AUG, leaky_random_split=leaky, honest_split_by_clip=honest, chance=1 / len(np.unique(src_y))),
)
Path("report.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k not in ("per_cat", "confusion")}, indent=1))
print("confusion (rows=true, cols=pred)", classes.tolist(), report["confusion"])
