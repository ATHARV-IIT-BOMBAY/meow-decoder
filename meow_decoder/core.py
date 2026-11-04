"""Meow Decoder core: audio in, likely situations out.

Heavy libraries (torch, transformers, avex, sklearn) are imported inside functions so the CLI and
tests start fast and models load once, on first use.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
from pathlib import Path

import numpy as np

SR = 16000
HF_REPO = "AtharvIITB/meow-decoder"
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
BEATS_ID = "esp_aves2_sl_beats_all"
ENCODERS = {
    "beats": "Earth Species Project BEATs (esp_aves2_sl_beats_all), CC BY-NC-SA 4.0: non-commercial only",
    "ast": "MIT AST fine-tuned on AudioSet, BSD-3-Clause: commercial use OK",
}
CAT_LABELS = ["Cat", "Meow", "Purr", "Caterwaul", "Hiss"]
CAT_GATE = 0.05  # ~5th percentile of AST cat-sound prob on real CatMeows clips
CONTEXTS = {  # key -> (emoji, title, what the cat might be saying)
    "F": ("🍗", "Food, please", "I'm hungry. Where's my food?"),
    "B": ("😽", "Enjoying the attention", "That feels nice, keep going."),
    "I": ("🙀", "Stressed somewhere new", "I don't know this place. Where are you?"),
}


def describe(key: str) -> tuple[str, str, str]:
    return CONTEXTS.get(key, ("🐾", key, key))


# ---------- audio ----------

def decode(data: bytes, max_seconds: int = 30) -> np.ndarray:
    """Any audio/video bytes ffmpeg can read -> 16 kHz mono float32."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-t", str(max_seconds), "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                       input=data, capture_output=True)
    wave = np.frombuffer(r.stdout, np.float32)
    if r.returncode or len(wave) < SR // 10:
        raise ValueError("Couldn't read that audio. Try recording again or use a .wav/.mp3/.m4a file.")
    return wave


def load(path) -> np.ndarray:
    """Audio file -> 16 kHz mono float32 (librosa; what training and the benchmark use)."""
    import librosa
    return librosa.load(path, sr=SR, mono=True)[0]


# ---------- encoders ----------

@functools.cache
def _ast():
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ASTForAudioClassification.from_pretrained(AST_ID).to(dev).eval()
    return ASTFeatureExtractor.from_pretrained(AST_ID), model, dev, [model.config.label2id[l] for l in CAT_LABELS]


@functools.cache
def _beats():
    from avex import load_model
    return load_model(BEATS_ID, device="cpu", return_features_only=True).eval()


def analyze(wave: np.ndarray, encoder: str = "beats") -> tuple[np.ndarray, dict[str, float]]:
    """16 kHz wave -> (768-d embedding from `encoder`, AudioSet probability per cat sound)."""
    import torch
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {list(ENCODERS)}")
    fe, ast, dev, ids = _ast()
    with torch.no_grad():
        out = ast(**fe(wave, sampling_rate=SR, return_tensors="pt").to(dev), output_hidden_states=encoder == "ast")
        sounds = dict(zip(CAT_LABELS, torch.sigmoid(out.logits[0, ids]).tolist()))
        if encoder == "ast":
            h = ast.audio_spectrogram_transformer.layernorm(out.hidden_states[-1])
            emb = ((h[:, 0] + h[:, 1]) / 2).squeeze(0).cpu().numpy()  # AST pooling: mean of CLS + distill tokens
        else:
            emb = _beats()(torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[None]).mean(dim=1).squeeze(0).numpy()
    return emb, sounds


# ---------- classifier head ----------

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


class Head:
    """Standardize + multinomial logistic regression, stored as plain arrays (npz, no pickle).

    Keeps the training embeddings (X, y) so it can be refit with one cat's own meows.
    """
    C = 0.002  # strong L2: best held-out log-loss for both encoders

    def __init__(self, classes, mean, scale, coef, intercept, X=None, y=None, encoder="beats"):
        self.classes, self.mean, self.scale, self.coef, self.intercept = map(np.asarray, (classes, mean, scale, coef, intercept))
        self.X, self.y, self.encoder = X, y, encoder

    @classmethod
    def fit(cls, X, y, weight=None, encoder="beats") -> "Head":
        from sklearn.linear_model import LogisticRegression
        X, y = np.asarray(X, np.float32), np.asarray(y)
        mean, scale = X.mean(0), X.std(0)
        scale[scale == 0] = 1
        lr = LogisticRegression(C=cls.C, max_iter=3000).fit((X - mean) / scale, y, sample_weight=weight)
        if len(lr.classes_) < 3:
            raise ValueError("Need at least 3 situations to train.")
        return cls(lr.classes_, mean, scale, lr.coef_, lr.intercept_, X, y, encoder)

    def proba(self, X) -> np.ndarray:
        z = ((np.atleast_2d(X) - self.mean) / self.scale) @ self.coef.T + self.intercept
        return softmax(z)

    def personalized(self, samples: list[tuple[np.ndarray, str]], weight: float = 20) -> "Head":
        """Refit on the base meows plus one cat's (embedding, label) pairs, each weighted `weight` times.
        3 meows per situation took unseen-cat accuracy from 66% to 80% in the benchmark."""
        if not samples:
            return self
        X = np.vstack([self.X, [e for e, _ in samples]])
        y = np.r_[self.y, [l for _, l in samples]]
        w = np.r_[np.ones(len(self.y)), np.full(len(samples), weight)]
        return Head.fit(X, y, w, self.encoder)

    def save(self, path):
        np.savez_compressed(path, classes=self.classes, mean=self.mean, scale=self.scale, coef=self.coef,
                            intercept=self.intercept, X=self.X, y=self.y, encoder=self.encoder)

    @classmethod
    def load(cls, path) -> "Head":
        d = np.load(path, allow_pickle=False)
        return cls(d["classes"], d["mean"], d["scale"], d["coef"], d["intercept"], d["X"], d["y"], str(d["encoder"]))


def weights_path(encoder: str) -> Path:
    """Local weights/<encoder>.npz (from `meow train`) if present, else the published ones on Hugging Face."""
    local = Path(os.environ.get("MEOW_WEIGHTS", "weights")) / f"{encoder}.npz"
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, f"{encoder}.npz"))


# ---------- public API ----------

class Decoder:
    """>>> Decoder().predict("meow.wav")["results"][0]["title"]"""

    def __init__(self, encoder: str = "beats", weights=None):
        self.encoder = encoder
        self.head = Head.load(weights or weights_path(encoder))
        if self.head.encoder != encoder:
            raise ValueError(f"weights were trained for the {self.head.encoder!r} encoder, not {encoder!r}")

    def embed(self, audio) -> tuple[np.ndarray, dict[str, float]]:
        """audio: path, raw bytes, or 16 kHz mono float array."""
        if isinstance(audio, (str, Path)):
            audio = Path(audio).read_bytes()
        wave = decode(audio) if isinstance(audio, bytes) else np.asarray(audio, np.float32)
        return analyze(wave, self.encoder)

    def rank(self, emb, sounds, head: Head | None = None) -> dict:
        h = head or self.head
        p = h.proba(emb)[0]
        results = [dict(zip(["key", "emoji", "title", "line"], [k, *describe(k)]), p=round(float(v), 4))
                   for k, v in sorted(zip(h.classes.tolist(), p), key=lambda kv: -kv[1])]
        return {"is_cat": max(sounds.values()) >= CAT_GATE, "sounds": {k: round(v, 3) for k, v in sounds.items()}, "results": results}

    def predict(self, audio, head: Head | None = None) -> dict:
        return self.rank(*self.embed(audio), head)


# ---------- training & benchmark (CatMeows) ----------

def catmeows(data_dir) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """CatMeows filenames encode situation (first letter) and cat id (second field)."""
    files = sorted(Path(data_dir).glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files in {data_dir}. See the README for the CatMeows download.")
    return files, np.array([f.name[0] for f in files]), np.array([f.name.split("_")[1] for f in files])


def embed_files(files, encoder: str) -> np.ndarray:
    return np.stack([analyze(load(f), encoder)[0] for f in files])


SPLITS = Path(__file__).resolve().parent / "splits.json"  # cat id -> fold, shipped with the package


def benchmark(X, y, cats, splits=None) -> dict:
    """The unseen-cat protocol: 5 fixed folds of whole cats (benchmark/splits.json)."""
    from sklearn.metrics import log_loss
    fold_of = json.loads(Path(splits or SPLITS).read_text())
    folds = np.array([fold_of[c] for c in cats])
    P = np.zeros((len(y), len(np.unique(y))))
    for f in np.unique(folds):
        head = Head.fit(X[folds != f], y[folds != f])
        P[folds == f] = head.proba(X[folds == f])
    classes = np.unique(y)
    conf, hit = P.max(1), classes[P.argmax(1)] == y
    bins = np.digitize(conf, [0.5, 0.6, 0.7, 0.8])
    ece = sum(abs(conf[bins == b].mean() - hit[bins == b].mean()) * (bins == b).sum() for b in np.unique(bins)) / len(y)
    top2 = np.mean([t in classes[np.argsort(p)[-2:]] for t, p in zip(y, P)])
    return {"top1": float(hit.mean()), "top2": float(top2), "log_loss": float(log_loss(y, P, labels=classes)), "ece": float(ece),
            "majority": float(max(np.mean(y == c) for c in classes)), "clips": int(len(y)), "cats": int(len(np.unique(cats)))}
