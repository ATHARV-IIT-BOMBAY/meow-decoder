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
# Purring, hissing and yowling. Purring uses AudioSet's score (0.3: 0.2% false alarms on CatMeows' ordinary meows,
# 26 of 30 real purrs caught). With the default BEATs encoder, hissing and yowling come from detectors trained on
# FSD50K and Freesound recordings (meow_decoder/detectors.npz, benchmark/detectors.py): on the real-world test they
# caught 9 of 13 hisses (AudioSet 5) and 22 of 24 fights and mating calls (AudioSet 12). The AudioSet thresholds
# below are used in AST mode: hiss 0.01 (AudioSet scores even clear hisses about 0.02), yowl 0.3.
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


def sounds_in(wave: np.ndarray) -> dict:
    """AudioSet probability of each cat sound and of speech. AST hears 10.24 s at a time, so this is the
    maximum over consecutive 10 s windows of the whole clip. "_audioset" holds all 527 classes (for the AST-mode
    yowl detector); keys starting with "_" are internal and never returned to callers."""
    import torch

    fe, ast, dev, ids = _ast()
    windows = [wave[i : i + 10 * SR] for i in range(0, max(len(wave) - SR // 2, 1), 10 * SR)]  # skip a last sliver under 0.5 s
    with torch.no_grad():
        p = torch.cat([ast(**fe(w, sampling_rate=SR, return_tensors="pt").to(dev)).logits for w in windows]).sigmoid().max(0).values
    p = p.cpu().numpy()
    return {**dict(zip(CAT_LABELS + [SPEECH], p[ids].tolist())), "_audioset": p}


def analyze(wave: np.ndarray, encoder: str = "beats") -> tuple[np.ndarray, dict[str, float]]:
    """16 kHz wave -> (embedding, sound probabilities over the whole clip)."""
    return embedding(wave, encoder), sounds_in(wave)


# ---------- pitch contour ----------
# Meowsic (Schötz et al., Lund University): food-related meows tend to rise in pitch, vet-related ones to fall.
# Contour shape survives a change of microphone and room far better than the audio embedding does
# (benchmark/probes/README.md), so it gives the situation guess for cats nobody has taught yet.

PITCH_FEATURES = ["rise_st", "slope_st_per_s", "peak_pos"]
PITCH_MODEL = Path(__file__).with_name("pitch.npz")  # built by benchmark/pitch.py from CatMeows
DETECTORS = {  # hiss/yowl detectors on each encoder's embedding: benchmark/detectors.py
    "beats": Path(__file__).with_name("detectors.npz"),
    "ast": Path(__file__).with_name("detectors_ast.npz"),  # trained only on CC0 / CC BY audio, so commercial use stays open
}


def segments(wave: np.ndarray, sr: int = SR, max_len: float = 4.0) -> list[tuple[int, int]]:
    """Sound events (start, end) in samples, loudest first; the whole clip when nothing stands out.

    A frame is active when it is 10 dB above the clip's noise floor (10th percentile) and within 35 dB of
    its peak. Gaps under 150 ms are merged, events under 0.3 s dropped, 100 ms of padding added, and
    events longer than `max_len` cut to the loudest `max_len` seconds.
    """
    hop, win = sr // 100, sr // 40
    if len(wave) < win:
        return [(0, len(wave))]
    frames = np.lib.stride_tricks.sliding_window_view(wave, win)[::hop]
    db = 10 * np.log10((frames.astype(np.float64) ** 2).mean(1) + 1e-10)
    floor, peak = np.percentile(db, 10), db.max()
    if peak - floor < 10:
        return [(0, len(wave))]
    active = db > max(floor + 10, peak - 35)
    runs, start = [], None
    for i, a in enumerate(np.r_[active, False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if runs and start - runs[-1][1] < 15:  # merge gaps under 150 ms
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((start, i))
            start = None
    events = []
    for s, e in runs:
        if e - s < 30:  # under 0.3 s
            continue
        n = int(max_len * 100)
        if e - s > n:  # keep the loudest max_len seconds
            sums = np.convolve(db[s:e] - floor, np.ones(n), "valid")
            s = s + int(sums.argmax())
            e = s + n
        a, b = max(0, s * hop - sr // 10), min(len(wave), e * hop + win + sr // 10)
        events.append((a, b, float(db[s:e].max())))
    if not events:
        return [(0, len(wave))]
    return [(a, b) for a, b, _ in sorted(events, key=lambda t: -t[2])]


def pitch_track(
    wave: np.ndarray, sr: int = SR, fmin: float = 200, fmax: float = 2000, hop: int = 160, frame: int = 1024, threshold: float = 0.15
) -> np.ndarray:
    """F0 in Hz every `hop` samples (NaN where unvoiced), by YIN (de Cheveigné & Kawahara, 2002)."""
    if len(wave) < frame:
        return np.full(0, np.nan)
    x = np.lib.stride_tricks.sliding_window_view(wave.astype(np.float64), frame)[::hop]
    tmax, tmin, n = int(sr / fmin), int(sr / fmax), frame - int(sr / fmin)
    d = np.stack([((x[:, :n] - x[:, t : t + n]) ** 2).sum(1) for t in range(1, tmax + 1)], 1)  # d(tau), tau = 1..tmax
    cmnd = d * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d, 1), 1e-12)
    level = x.std(1)
    loud = level > max(1e-4, 0.03 * level.max())  # silence and faint background have no pitch worth tracking
    f0 = np.full(len(x), np.nan)
    for i, row in enumerate(cmnd):
        if not loud[i]:
            continue
        below = np.where(row[tmin - 1 :] < threshold)[0]
        if not len(below):
            continue
        t = below[0] + tmin - 1
        while t + 1 < tmax and row[t + 1] < row[t]:  # walk down to the local minimum
            t += 1
        a, b, c = row[max(t - 1, 0)], row[t], row[min(t + 1, tmax - 1)]
        shift = float(np.clip(0.5 * (a - c) / (a - 2 * b + c), -1, 1)) if a - 2 * b + c > 0 else 0.0  # parabolic interpolation
        f0[i] = sr / (t + 1 + shift)
    return f0


def contour(event: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """One sound event's pitch in semitones (re A4) and its times in seconds, cleaned; None when too little is voiced."""
    f0 = pitch_track(event)
    voiced = ~np.isnan(f0)
    if voiced.sum() < 8:
        return None
    st, t = 12 * np.log2(f0[voiced] / 440), np.where(voiced)[0] * 0.01
    keep = np.abs(st - np.median(st)) < 7  # drop octave errors: a meow doesn't jump an octave in 10 ms
    st, t = st[keep], t[keep]
    if len(st) < 8:
        return None
    return np.convolve(np.pad(st, 2, mode="edge"), np.ones(5) / 5, "valid"), t  # smooth tracker jitter over 50 ms


def pitch_features(wave: np.ndarray) -> np.ndarray:
    """Contour shape of the loudest three sound events, averaged: rise (last third minus first third, semitones),
    slope (semitones per second) and where the peak falls (0 = start, 1 = end). NaN when no meow pitch is found."""
    rows = []
    for a, b in segments(wave)[:3]:
        c = contour(wave[a:b])
        if c is None:
            continue
        st, t = c
        third = max(len(st) // 3, 1)
        rows.append([st[-third:].mean() - st[:third].mean(), np.polyfit(t, st, 1)[0], float(np.argmax(st)) / len(st)])
    return np.mean(rows, 0) if rows else np.full(len(PITCH_FEATURES), np.nan)


def fit_pitch_head(F: np.ndarray, y: np.ndarray) -> Head:
    """Logistic regression on contour shape, classes weighted equally (CatMeows is half "stressed" meows).
    Missing pitch is filled with the training mean, so a clip with no pitch gets a near-even guess."""
    from sklearn.linear_model import LogisticRegression

    mean, scale = np.nanmean(F, 0), np.nanstd(F, 0)
    Z = np.nan_to_num((F - mean) / scale)
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(Z, y)
    return Head(lr.classes_, mean, scale, lr.coef_, lr.intercept_, F, np.asarray(y), "pitch")


def pitch_line(wave: np.ndarray, points: int = 48) -> dict | None:
    """The loudest meow's pitch for drawing: semitones relative to its start, at most `points` values, and its rise."""
    for a, b in segments(wave)[:3]:
        c = contour(wave[a:b])
        if c is not None:
            st = c[0] - c[0][0]
            idx = np.linspace(0, len(st) - 1, min(points, len(st))).round().astype(int)
            third = max(len(st) // 3, 1)
            return {"semitones": [round(float(x), 2) for x in st[idx]], "seconds": round(len(st) * 0.01, 2),
                    "rise": round(float(st[-third:].mean() - st[:third].mean()), 2)}  # fmt: skip
    return None


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

    def proba_features(self, F) -> np.ndarray:
        """proba() for pitch features, which may be NaN (no pitch found): missing values count as average."""
        F = np.atleast_2d(np.asarray(F, np.float64))
        return self.proba(np.where(np.isnan(F), self.mean, F))

    def personalized(self, samples: list[tuple[np.ndarray, str]], weight: float = 20) -> Head:
        """Refit on the base meows plus one cat's (embedding, label) pairs, each weighted `weight` times.
        3 meows per situation took unseen-cat accuracy from 65% to 72% on another recording session."""
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
        self.pitch_head = Head.load(PITCH_MODEL) if PITCH_MODEL.exists() else None
        self.detectors = dict(np.load(DETECTORS[encoder])) if DETECTORS[encoder].exists() else None

    def wave(self, audio) -> np.ndarray:
        """audio: path, raw bytes, or 16 kHz mono float array."""
        if isinstance(audio, (str, Path)):
            audio = Path(audio).read_bytes()
        return decode(audio) if isinstance(audio, bytes) else np.asarray(audio, np.float32)

    def embed(self, audio) -> tuple[np.ndarray, dict[str, float]]:
        """audio: path, raw bytes, or 16 kHz mono float array."""
        return analyze(self.wave(audio), self.encoder)

    def rank(self, emb, sounds, head: Head | None = None, pitch=None) -> dict:
        """Situation guess from `head` (a cat's taught head); without one, from the pitch contour when given,
        else from the embedding model, which carries over poorly to home recordings (README)."""
        if head is None and pitch is not None and self.pitch_head is not None:
            h, p, source = self.pitch_head, self.pitch_head.proba_features(pitch)[0], "pitch"
        else:
            h = head or self.head
            p, source = h.proba(emb)[0], "taught" if head is not None and head is not self.head else "embedding"
        results = [
            dict(zip(["key", "emoji", "title", "line"], [k, *describe(k)]), p=round(float(v), 4))
            for k, v in sorted(zip(h.classes.tolist(), p), key=lambda kv: -kv[1])
        ]
        signals = [
            dict(key=k, emoji=e, title=t, meaning=m, p=round(p, 3))
            for k, (e, t, m, _) in SIGNALS.items()
            if (p := self.heard(k, emb, sounds))
        ]
        # Not "or a detector fired": that found 2 more real cats but called 3 more people imitating cats a cat.
        is_cat = max(sounds[k] for k in CAT_LABELS) >= CAT_GATE
        # The situation model only knows meows: with no cat, or a purr or hiss with no meow in it, its guess means nothing.
        has_meow = is_cat and (sounds["Meow"] >= MEOW_MIN or not signals)
        return {
            "is_cat": is_cat,
            "sounds": {k: round(v, 3) for k, v in sounds.items() if not k.startswith("_")},
            "signals": signals,
            "has_meow": has_meow,
            "results": results,
            "guess_from": source,
        }

    def heard(self, key: str, emb, sounds) -> float:
        """How sure it is that `key` (Purr, Hiss, Caterwaul) is in the clip, or 0 when it isn't. The BEATs detectors
        (benchmark/detectors.py) caught 12 of 13 real hisses where AudioSet's score caught 5; AST mode keeps AudioSet's."""
        d = getattr(self, "detectors", None)
        if d is None or f"{key}_coef" not in d:
            return sounds[key] if sounds[key] >= SIGNALS[key][3] else 0.0
        x = np.asarray(emb)
        if str(d.get("features", "embedding")) == "audioset":  # AST mode: all 527 AudioSet scores, as log-odds
            p = np.clip(sounds["_audioset"], 1e-6, 1 - 1e-6)
            x = np.log(p / (1 - p))
        z = float((x - d[f"{key}_mean"]) / d[f"{key}_scale"] @ d[f"{key}_coef"] + d[f"{key}_intercept"])
        margin = z - float(d[f"{key}_threshold"])
        return 1 / (1 + np.exp(-margin)) if margin > 0 else 0.0

    def predict(self, audio, head: Head | None = None) -> dict:
        wave = self.wave(audio)
        return self.rank(*analyze(wave, self.encoder), head, pitch_features(wave))
