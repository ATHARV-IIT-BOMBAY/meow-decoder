"""Evaluation: the fixed unseen-cat benchmark, teaching across recording sessions, and robustness.

Every score here comes from cats the model never trained on. Teaching is scored on recordings from a
different session than the taught meows, because meows from one session share the room and microphone.
"""

from __future__ import annotations

import json
import subprocess
import zlib
from pathlib import Path

import numpy as np

from .core import SR, Head, decode, embedding, load, segments  # noqa: F401  (segments re-exported)

AUGMENT = 0  # augmented training copies per clip (see augment()); set from benchmark results

SPLITS = Path(__file__).resolve().parent / "splits.json"  # cat id -> fold, shipped with the package


def catmeows(data_dir) -> tuple[list[Path], np.ndarray, np.ndarray, np.ndarray]:
    """CatMeows filenames C_NNNNN_BB_SS_OOOOO_RXX encode situation, cat id and recording session R."""
    files = sorted(Path(data_dir).glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files in {data_dir}. See the README for the CatMeows download.")
    parts = [f.stem.split("_") for f in files]
    return files, np.array([f.name[0] for f in files]), np.array([p[1] for p in parts]), np.array([p[-1][0] for p in parts])


def folds_for(cats) -> np.ndarray:
    fold_of = json.loads(SPLITS.read_text())
    return np.array([fold_of[c] for c in cats])


def unseen_cat_proba(X, y, cats, train_only=None, weight=None) -> np.ndarray:
    """Out-of-fold probabilities over the fixed whole-cat folds. Rows marked train_only (e.g. augmented
    copies) are only ever used for training; their returned rows are left at zero."""
    folds, classes = folds_for(cats), np.unique(y)
    test_rows = np.ones(len(y), bool) if train_only is None else ~np.asarray(train_only, bool)
    P = np.zeros((len(y), len(classes)))
    for f in np.unique(folds):
        h = Head.fit(X[folds != f], y[folds != f], None if weight is None else weight[folds != f])
        te = (folds == f) & test_rows
        P[te] = h.proba(X[te])[:, np.searchsorted(h.classes, classes)]
    return P


def metrics(P, y, cats, resamples: int = 2000) -> dict:
    """Accuracy, top two, macro recall, log-loss, calibration error, and a 95% interval from resampling cats."""
    classes = np.unique(y)
    hit = classes[P.argmax(1)] == y
    conf = P.max(1)
    bins = np.digitize(conf, [0.5, 0.6, 0.7, 0.8])
    ece = sum(abs(conf[bins == b].mean() - hit[bins == b].mean()) * (bins == b).sum() for b in np.unique(bins)) / len(y)
    top2 = np.mean([t in classes[np.argsort(p)[-2:]] for t, p in zip(y, P)])
    rows = {c: np.where(cats == c)[0] for c in np.unique(cats)}
    rng = np.random.default_rng(0)
    boot = [hit[np.concatenate([rows[c] for c in rng.choice(list(rows), len(rows))])].mean() for _ in range(resamples)] or [hit.mean()]
    p = np.clip(P[np.arange(len(y)), np.searchsorted(classes, y)], 1e-12, 1)
    return {
        "top1": float(hit.mean()),
        "top1_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "top2": float(top2),
        "macro_recall": float(np.mean([hit[y == c].mean() for c in classes])),
        "log_loss": float(-np.log(p).mean()),
        "ece": float(ece),
        "confidence": float(conf.mean()),
        "recall": {str(c): float(hit[y == c].mean()) for c in classes},
        "majority": float(max(np.mean(y == c) for c in classes)),
        "clips": int(len(y)),
        "cats": int(len(rows)),
    }


def benchmark(X, y, cats, train_only=None) -> dict:
    """The unseen-cat benchmark. Rows marked train_only (augmented copies) help train but are never scored."""
    P = unseen_cat_proba(X, y, cats, train_only)
    m = np.ones(len(y), bool) if train_only is None else ~np.asarray(train_only, bool)
    return metrics(P[m], y[m], cats[m])


def teaching_curve(X, y, cats, sessions, ks=(0, 1, 2, 3, 4), weight: float = 20, same_session: bool = False, only=None) -> list[dict]:
    """Accuracy on one cat after teaching it k meows per situation.

    The taught meows all come from one recording session. By default the cat is then tested on its other
    sessions (a different day, like real use); same_session=True tests on the rest of the teaching session,
    which is optimistic. Each cat takes every session in turn as the teaching session. `only` limits
    which cats are scored (all others still train the base model).
    """
    out = []
    for k in ks:
        hits, confs = [], []
        for cat in np.unique(cats) if only is None else sorted(only):
            mine = np.where(cats == cat)[0]
            others = np.where(cats != cat)[0]
            for s in np.unique(sessions[mine]):
                in_s = mine[sessions[mine] == s]
                test_pool = in_s if same_session else mine[sessions[mine] != s]
                if not len(test_pool):
                    continue
                rng = np.random.default_rng(zlib.crc32(f"{cat}/{s}/{k}".encode()))
                teach = []
                for lab in np.unique(y[in_s]):
                    pool = in_s[y[in_s] == lab]
                    take = min(k, len(pool) - 1) if same_session else min(k, len(pool))
                    teach += list(rng.choice(pool, max(take, 0), replace=False))
                test = np.setdiff1d(test_pool, teach)
                if not len(test):
                    continue
                tr = np.r_[others, teach].astype(int)
                w = np.r_[np.ones(len(others)), np.full(len(teach), weight)]
                h = Head.fit(X[tr], y[tr], w)
                P = h.proba(X[test])
                hits += list(h.classes[P.argmax(1)] == y[test])
                confs += list(P.max(1))
        out.append({"k": k, "accuracy": float(np.mean(hits)), "confidence": float(np.mean(confs)), "tested": len(hits)})
    return out


# ---------- robustness: what the app actually hears ----------

# Clean-up steps suggested in review (find the meow, match the 8 kHz training band, fix the level).
# Tested on the robustness grid and NOT used by the app: each lowered clean accuracy and none raised the
# average over real-world conditions (see README, "What we tried that didn't help").


def band_limit(wave: np.ndarray, sr: int = SR, cutoff: float = 4000.0) -> np.ndarray:
    """Low-pass to `cutoff` Hz (smooth 400 Hz roll-off) so phone audio matches the 8 kHz training recordings."""
    spec = np.fft.rfft(wave)
    f = np.fft.rfftfreq(len(wave), 1 / sr)
    spec *= np.clip((cutoff - f) / 400 + 0.5, 0, 1)
    return np.fft.irfft(spec, len(wave)).astype(np.float32)


def normalize(wave: np.ndarray, rms: float = 0.05) -> np.ndarray:
    """Scale to a fixed loudness (RMS), clipped to [-1, 1]."""
    r = float(np.sqrt((wave.astype(np.float64) ** 2).mean()))
    return np.clip(wave * (rms / r), -1, 1).astype(np.float32) if r > 1e-6 else wave


def opus(w: np.ndarray) -> np.ndarray:
    """Round trip through 48 kHz Opus in WebM, what a browser microphone recording goes through."""
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(SR),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-ar",
            "48000",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-f",
            "webm",
            "pipe:1",
        ],
        input=np.asarray(w, np.float32).tobytes(),
        capture_output=True,
    )
    return decode(r.stdout)


def add_noise(w, snr_db, rng):
    n = rng.normal(0, 1, len(w)).astype(np.float32)
    return (w + n * np.sqrt((w**2).mean() / (n**2).mean() / 10 ** (snr_db / 10))).astype(np.float32)


def in_recording(w, rng, seconds: float = 8.0):
    """The meow somewhere inside a longer recording with faint room noise."""
    room = (rng.normal(0, 1, int(SR * seconds)) * 0.002).astype(np.float32)
    s = int(rng.uniform(0.5, seconds - 0.5 - len(w) / SR) * SR) if seconds * SR > len(w) + SR else 0
    room[s : s + len(w)] += w[: len(room) - s]
    return room


CONDITIONS = {  # name -> (description, f(wave, rng))
    "opus": ("Browser microphone codec (Opus)", lambda w, r: opus(w)),
    "quiet": ("10x quieter", lambda w, r: w * 0.1),
    "loud": ("3x louder, clipped", lambda w, r: np.clip(w * 3, -1, 1)),
    "noise20": ("White noise, 20 dB SNR", lambda w, r: add_noise(w, 20, r)),
    "noise10": ("White noise, 10 dB SNR", lambda w, r: add_noise(w, 10, r)),
    "in8s": ("Meow inside an 8 s recording", lambda w, r: in_recording(w, r)),
}


def augment_copy(w, k: int, rng) -> np.ndarray:
    """Training copies: (0) Opus, (1) noise at 10-30 dB, (2) inside a 3-8 s recording; each with a random gain."""
    g = np.clip(w * 10 ** (rng.uniform(-12, 6) / 20), -1, 1).astype(np.float32)
    if k == 0:
        return opus(g)
    if k == 1:
        return add_noise(g, rng.uniform(10, 30), rng)
    return in_recording(g, rng, rng.uniform(3, 8))


def embed_dataset(files, y, cats, encoder: str, augment: int | None = None):
    """Embeddings for training: each clip, plus `augment` noisy/recompressed/padded copies marked train-only."""
    augment = AUGMENT if augment is None else augment
    waves = [load(f) for f in files]
    X = [embedding(w, encoder) for w in waves]
    yy, cc, extra = list(y), list(cats), [False] * len(y)
    for k in range(augment):
        for i, w in enumerate(waves):
            X.append(embedding(augment_copy(w, k, np.random.default_rng(zlib.crc32(f"aug/{i}/{k}".encode()))), encoder))
            yy.append(y[i])
            cc.append(cats[i])
            extra.append(True)
    return np.stack(X), np.array(yy), np.array(cc), np.array(extra)


def robustness(files, y, cats, encoder: str) -> dict:
    """Accuracy on unseen cats when their audio goes through each real-world condition. Training is exactly
    what `meow train` does, on the other folds."""
    X, yy, cc, train_only = embed_dataset(files, y, cats, encoder)
    folds_all, folds = folds_for(cc), folds_for(cats)
    heads = {f: Head.fit(X[folds_all != f], yy[folds_all != f]) for f in np.unique(folds)}
    waves = [load(f) for f in files]
    out = {}
    for name, (_, change) in CONDITIONS.items():
        Z = np.stack([embedding(change(w, np.random.default_rng(1000 + i)), encoder) for i, w in enumerate(waves)])
        classes = np.unique(y)
        P = np.zeros((len(y), len(classes)))
        for f, h in heads.items():
            P[folds == f] = h.proba(Z[folds == f])[:, np.searchsorted(h.classes, classes)]
        out[name] = metrics(P, y, cats, resamples=0)
    return out
