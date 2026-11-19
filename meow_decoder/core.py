"""Meow Decoder core: audio in, likely situations out.

Heavy libraries (torch, transformers, avex, sklearn) are imported inside functions so the CLI and
tests start fast and models load once, on first use.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

import numpy as np

SR = 16000
HF_REPO = "AtharvIITB/meow-decoder"
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
BEATS_ID = "esp_aves2_sl_beats_all"
# Exact revisions, so an install made today keeps getting the same models and weights.
AST_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
BEATS_REPO, BEATS_FILE = "EarthSpeciesProject/esp-aves2-sl-beats-all", "esp-aves2-sl-beats-all.safetensors"
BEATS_REVISION = "054a6466cfc8caa5dfa079578f013689bbf751c2"
WEIGHTS_REVISION = "96909777e22a78b23e1000db257466496b52be8f"
ENCODERS = {
    "beats": "Earth Species Project BEATs (esp_aves2_sl_beats_all), CC BY-NC-SA 4.0: non-commercial only",
    "ast": "MIT AST fine-tuned on AudioSet, BSD-3-Clause: commercial use OK",
}
CAT_LABELS = ["Cat", "Meow", "Purr", "Caterwaul", "Hiss"]
SPEECH = "Speech"  # also read from AudioSet, to keep talking out of the open dataset
CAT_GATE = 0.05  # ~5th percentile of AST cat-sound prob on real CatMeows clips
MEOW_MIN = 0.3  # AST "Meow" score reached by 81% of CatMeows meows
CONTEXTS = {  # key -> (emoji, title, what the cat might be saying); learned from CatMeows for every cat
    "F": ("🍗", "Food, please", "I'm hungry. Where's my food?"),
    "B": ("😽", "Enjoying the attention", "That feels nice, keep going."),
    "I": ("🙀", "Stressed somewhere new", "I don't know this place. Where are you?"),
}
# Situations you can teach your own cat. No open data covers them yet, so they only exist for cats you teach;
# shared meows with these labels are how they can become available to everyone.
PRESETS = {
    "greet": ("👋", "Saying hello", "Oh, you're back!"),
    "out": ("🚪", "Wants to go out", "Open the door, please."),
    "play": ("🧶", "Wants to play", "Play with me!"),
    "annoyed": ("😾", "Annoyed", "Leave me alone."),
}
# Sounds the AudioSet model recognises directly. Thresholds are set on CatMeows' 440 ordinary meows
# (benchmark/sound_types.py) and checked on real recordings from the internet (benchmark/wild.py):
# purr 0.3: 0.2% false alarms on ordinary meows; caught 14 of 18 real purrs, 0 false alarms on 38 other clips.
# hiss 0.01: 0.2% false alarms on ordinary meows. AudioSet scores even clear hisses low (about 0.02), so the
#   earlier 0.2 never fired; 0.01 caught 3 of 4 real hisses with no false alarms (revised after seeing them).
# yowl 0.3: 3% false alarms on ordinary meows; caught 4 of 8 real fights, and fires on some loud demanding meows.
SIGNALS = {  # AudioSet label -> (emoji, title, meaning, threshold)
    "Purr": (
        "😌",
        "Purring",
        "Usually relaxed and content. Cats also purr to calm themselves when stressed or hurt, so check the rest of its body language.",
        0.3,
    ),
    "Hiss": ("😾", "Hissing", "Scared or defensive. Give it space and don't reach for it.", 0.01),
    "Caterwaul": (
        "📢",
        "Yowling",
        "A long, intense call. Often a standoff with another cat or a mating call; it can also mean distress.",
        0.3,
    ),
}


def describe(key: str) -> tuple[str, str, str]:
    return CONTEXTS.get(key) or PRESETS.get(key) or ("🐾", key, key)


# ---------- audio ----------


def decode(data: bytes, max_seconds: int = 30) -> np.ndarray:
    """Any audio/video bytes ffmpeg can read -> 16 kHz mono float32.

    Safe for untrusted uploads: ffmpeg may only read the pipe (no files or URLs a playlist could point at),
    ignores video, and is killed after 20 seconds.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "pipe",
        "-i",
        "pipe:0",
        "-vn",
        "-t",
        str(max_seconds),
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        r = subprocess.run(cmd, input=data, capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        raise ValueError("That file took too long to read. Try a shorter recording.") from None
    except FileNotFoundError:
        raise ValueError("ffmpeg isn't installed. See the README's Install section.") from None
    wave = np.frombuffer(r.stdout, np.float32)
    if r.returncode or len(wave) < SR // 10:
        raise ValueError("Couldn't read that audio. Try recording again or use a .wav/.mp3/.m4a file.")
    return wave


def load(path) -> np.ndarray:
    """Audio file -> 16 kHz mono float32 (librosa; what training and the benchmark use)."""
    try:
        import librosa
    except ImportError:
        raise ValueError("Training and benchmarks need librosa: pip install 'meow-decoder[train]'") from None
    return librosa.load(path, sr=SR, mono=True)[0]


# ---------- encoders ----------


@functools.cache
def _ast():
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ASTForAudioClassification.from_pretrained(AST_ID, revision=AST_REVISION).to(dev).eval()
    ids = [model.config.label2id[name] for name in CAT_LABELS + [SPEECH]]
    return ASTFeatureExtractor.from_pretrained(AST_ID, revision=AST_REVISION), model, dev, ids


@functools.cache
def _beats():
    from avex import load_model
    from huggingface_hub import hf_hub_download

    checkpoint = hf_hub_download(BEATS_REPO, BEATS_FILE, revision=BEATS_REVISION)
    return load_model(BEATS_ID, device="cpu", return_features_only=True, checkpoint_path=checkpoint).eval()


def embedding(wave: np.ndarray, encoder: str = "beats") -> np.ndarray:
    """768 numbers describing the meow, from `encoder`."""
    import torch

    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {list(ENCODERS)}")
    heard = np.ascontiguousarray(wave, dtype=np.float32)
    with torch.no_grad():
        if encoder == "beats":
            return _beats()(torch.from_numpy(heard)[None]).mean(dim=1).squeeze(0).numpy()
        fe, ast, dev, _ = _ast()
        h = ast.audio_spectrogram_transformer.layernorm(
            ast(**fe(heard, sampling_rate=SR, return_tensors="pt").to(dev), output_hidden_states=True).hidden_states[-1]
        )
        return ((h[:, 0] + h[:, 1]) / 2).squeeze(0).cpu().numpy()  # AST pooling: mean of CLS + distill tokens


def sounds_in(wave: np.ndarray) -> dict[str, float]:
    """AudioSet probability of each cat sound and of speech. AST hears 10.24 s at a time, so this is the
    maximum over consecutive 10 s windows of the whole clip."""
    import torch

    fe, ast, dev, ids = _ast()
    windows = [wave[i : i + 10 * SR] for i in range(0, max(len(wave) - SR // 2, 1), 10 * SR)]  # skip a last sliver under 0.5 s
    with torch.no_grad():
        logits = torch.cat([ast(**fe(w, sampling_rate=SR, return_tensors="pt").to(dev)).logits[:, ids] for w in windows])
    return dict(zip(CAT_LABELS + [SPEECH], torch.sigmoid(logits).max(0).values.tolist()))


def analyze(wave: np.ndarray, encoder: str = "beats") -> tuple[np.ndarray, dict[str, float]]:
    """16 kHz wave -> (embedding, sound probabilities over the whole clip)."""
    return embedding(wave, encoder), sounds_in(wave)


# ---------- classifier head ----------


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


class Head:
    """Standardize + multinomial logistic regression, stored as plain arrays (npz, no pickle).

    Keeps the training embeddings (X, y) so it can be refit with one cat's own meows.
    """

    C = 0.002  # strong L2: best held-out log-loss for both encoders

    def __init__(self, classes, mean, scale, coef, intercept, X=None, y=None, encoder="beats", cats=None, train_only=None):
        self.classes, self.mean, self.scale, self.coef, self.intercept = map(np.asarray, (classes, mean, scale, coef, intercept))
        self.X, self.y, self.encoder = X, y, encoder
        # Per row of X: which cat it came from, and whether it is an augmented copy (used only for training).
        # Stored so `meow benchmark --from-weights` can re-run the unseen-cat benchmark without audio or models.
        self.cats, self.train_only = cats, train_only

    @classmethod
    def fit(cls, X, y, weight=None, encoder="beats") -> Head:
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

    def personalized(self, samples: list[tuple[np.ndarray, str]], weight: float = 20) -> Head:
        """Refit on the base meows plus one cat's (embedding, label) pairs, each weighted `weight` times.
        3 meows per situation took unseen-cat accuracy from 66% to 80% in the benchmark."""
        if not samples:
            return self
        X = np.vstack([self.X, [e for e, _ in samples]])
        y = np.r_[self.y, [label for _, label in samples]]
        w = np.r_[np.ones(len(self.y)), np.full(len(samples), weight)]
        return Head.fit(X, y, w, self.encoder)

    def save(self, path):
        extra = {k: v for k, v in (("cats", self.cats), ("train_only", self.train_only)) if v is not None}
        np.savez_compressed(
            path,
            classes=self.classes,
            mean=self.mean,
            scale=self.scale,
            coef=self.coef,
            intercept=self.intercept,
            X=self.X,
            y=self.y,
            encoder=self.encoder,
            **extra,
        )

    @classmethod
    def load(cls, path) -> Head:
        d = np.load(path, allow_pickle=False)
        return cls(
            d["classes"],
            d["mean"],
            d["scale"],
            d["coef"],
            d["intercept"],
            d["X"],
            d["y"],
            str(d["encoder"]),
            d["cats"] if "cats" in d else None,
            d["train_only"] if "train_only" in d else None,
        )


def weights_path(encoder: str) -> Path:
    """Local weights/<encoder>.npz (from `meow train`) if present, else the published ones on Hugging Face."""
    local = Path(os.environ.get("MEOW_WEIGHTS", "weights")) / f"{encoder}.npz"
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, f"{encoder}.npz", revision=WEIGHTS_REVISION))


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
        results = [
            dict(zip(["key", "emoji", "title", "line"], [k, *describe(k)]), p=round(float(v), 4))
            for k, v in sorted(zip(h.classes.tolist(), p), key=lambda kv: -kv[1])
        ]
        signals = [
            dict(key=k, emoji=e, title=t, meaning=m, p=round(sounds[k], 3)) for k, (e, t, m, th) in SIGNALS.items() if sounds[k] >= th
        ]
        is_cat = max(sounds[k] for k in CAT_LABELS) >= CAT_GATE
        # The situation model only knows meows: with no cat, or a purr or hiss with no meow in it, its guess means nothing.
        has_meow = is_cat and (sounds["Meow"] >= MEOW_MIN or not signals)
        return {
            "is_cat": is_cat,
            "sounds": {k: round(v, 3) for k, v in sounds.items()},
            "signals": signals,
            "has_meow": has_meow,
            "results": results,
        }

    def predict(self, audio, head: Head | None = None) -> dict:
        return self.rank(*self.embed(audio), head)
