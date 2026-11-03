"""Cat meow intent classifier: frozen bioacoustic BEATs embeddings + logistic regression.
AST (AudioSet) only answers "is this a cat sound, and is it a purr/hiss/yowl?".

Run `python meow.py` to embed CatMeows, evaluate on held-out cats, and save model.joblib.
"""
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from avex import load_model
from transformers import ASTFeatureExtractor, ASTForAudioClassification

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
# Earth Species Project encoder (CC-BY-NC-SA: non-commercial). Beat AST on unseen cats: 66% vs 62%, log-loss 0.76 vs 0.89.
ENCODER_ID = "esp_aves2_sl_beats_all"
SR = 16000
CAT_LABELS = ["Cat", "Meow", "Purr", "Caterwaul", "Hiss"]
CONTEXTS = {  # key -> (emoji, title, what the cat might be saying)
    "F": ("🍗", "Food, please", "I'm hungry. Where's my food?"),
    "B": ("😽", "Enjoying the attention", "That feels nice, keep going."),
    "I": ("🙀", "Stressed somewhere new", "I don't know this place. Where are you?"),
}

device = "mps" if torch.backends.mps.is_available() else "cpu"
_fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
_ast = ASTForAudioClassification.from_pretrained(MODEL_ID).to(device).eval()
_cat_ids = [_ast.config.label2id[l] for l in CAT_LABELS]
_enc = load_model(ENCODER_ID, device="cpu", return_features_only=True).eval()


@torch.no_grad()
def analyze(wave: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """16 kHz mono wave -> (768-d BEATs embedding, AudioSet probability per cat sound in CAT_LABELS)."""
    logits = _ast(**_fe(wave, sampling_rate=SR, return_tensors="pt").to(device)).logits[0]
    emb = _enc(torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[None]).mean(dim=1).squeeze(0).numpy()  # mean over time
    return emb, dict(zip(CAT_LABELS, torch.sigmoid(logits[_cat_ids]).tolist()))


def load(path) -> np.ndarray:
    return librosa.load(path, sr=SR, mono=True)[0]



if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    files = sorted(Path("data/dataset").glob("*.wav"))
    y = np.array([f.name[0] for f in files])
    cats = np.array([f.name.split("_")[1] for f in files])
    X, cat_ps = zip(*(analyze(load(f)) for f in files))
    X, cat_ps = np.stack(X), np.array([max(s.values()) for s in cat_ps])
    print(f"{len(files)} clips, {len(set(cats))} cats, classes {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"cat-sound prob on real meows: min {cat_ps.min():.2f}, 5th pct {np.percentile(cat_ps, 5):.2f}, median {np.median(cat_ps):.2f}")

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.002, max_iter=3000))  # strong L2: calibrated on unseen cats
    chance = max(np.mean(y == c) for c in CONTEXTS)
    for name, cv, groups in [
        ("unseen cats (GroupKFold by cat)", GroupKFold(5), cats),
        ("seen cats (StratifiedKFold)", StratifiedKFold(5, shuffle=True, random_state=0), None),
    ]:
        P = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")
        classes = np.unique(y)
        print(f"{name}: acc {accuracy_score(y, classes[P.argmax(1)]):.2f} "
              f"(chance {chance:.2f}), log-loss {log_loss(y, P, labels=classes):.2f}")

    clf.fit(X, y)
    joblib.dump(clf, "model.joblib")
    np.savez("base.npz", X=X, y=y)  # app refits on these + a cat's own clips
    print("saved model.joblib, base.npz")
