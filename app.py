"""Meow Decoder server. Run: .venv/bin/python app.py  (then open http://localhost:8000)"""
import json
import subprocess
import uuid
from pathlib import Path

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.base import clone

from meow import CONTEXTS, SR, analyze

CAT_GATE = 0.05     # ~5th percentile of AST cat-sound prob on real CatMeows clips
TEACH_WEIGHT = 20   # sample weight for a cat's own clips; 3 clips/situation took unseen-cat acc 66% -> 80%
MAX_UPLOAD = 10 * 2**20
STORE = Path("data/mine/cats.json")

base_clf = joblib.load("model.joblib")
base = np.load("base.npz")
cats: dict = json.loads(STORE.read_text()) if STORE.exists() else {}
_models: dict = {}  # cat_id -> (n_samples, fitted clf)
_clips: dict = {}   # clip token -> embedding, so "teach" needn't re-upload


def save():
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(cats))


def model_for(cat_id):
    samples = cats.get(cat_id, {}).get("samples", [])
    if not samples:
        return base_clf
    if _models.get(cat_id, (None,))[0] != len(samples):
        X = np.vstack([base["X"], [s["emb"] for s in samples]])
        y = np.r_[base["y"], [s["label"] for s in samples]]
        w = np.r_[np.ones(len(base["y"])), np.full(len(samples), TEACH_WEIGHT)]
        _models[cat_id] = (len(samples), clone(base_clf).fit(X, y, logisticregression__sample_weight=w))
    return _models[cat_id][1]


def decode(data: bytes) -> np.ndarray:
    """Any browser recording (webm/ogg/mp4/wav...) -> 16 kHz mono float32, max 30 s."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-t", "30", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                       input=data, capture_output=True)
    wave = np.frombuffer(r.stdout, np.float32)
    if r.returncode or len(wave) < SR // 10:
        raise HTTPException(400, "Couldn't read that audio. Try recording again or upload a .wav/.mp3/.m4a file.")
    return wave


def describe(key):
    return CONTEXTS.get(key, ("🐾", key, key))


def public(cid, c):
    counts = {}
    for s in c["samples"]:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    return {"id": cid, "name": c["name"], "photo": c.get("photo"), "counts": counts}


app = FastAPI()


@app.get("/api/cats")
def list_cats():
    return [public(k, c) for k, c in cats.items()]


@app.post("/api/cats")
def add_cat(name: str = Form(..., max_length=40), photo: str = Form("", max_length=300_000)):
    if not name.strip():
        raise HTTPException(400, "Give your cat a name.")
    if photo and not photo.startswith("data:image/"):
        raise HTTPException(400, "Photo must be an image.")
    cid = uuid.uuid4().hex[:8]
    cats[cid] = {"name": name.strip(), "photo": photo or None, "samples": []}
    save()
    return public(cid, cats[cid])


@app.delete("/api/cats/{cid}")
def delete_cat(cid: str):
    cats.pop(cid, None)
    _models.pop(cid, None)
    save()
    return {"ok": True}


@app.post("/api/predict")
async def predict(audio: UploadFile, cat_id: str = Form("")):
    data = await audio.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That file is over 10 MB. Trim it to a few seconds of meowing.")
    emb, sounds = analyze(decode(data))
    clf = model_for(cat_id)
    probs = sorted(zip(clf.classes_, clf.predict_proba([emb])[0]), key=lambda kv: -kv[1])
    token = uuid.uuid4().hex
    _clips[token] = emb
    if len(_clips) > 200:
        _clips.pop(next(iter(_clips)))
    return {
        "clip": token,
        "is_cat": max(sounds.values()) >= CAT_GATE,
        "sounds": {k: round(v, 3) for k, v in sounds.items()},
        "taught": len(cats.get(cat_id, {}).get("samples", [])),
        "results": [dict(zip(["key", "emoji", "title", "line"], [k, *describe(k)]), p=round(float(p), 4)) for k, p in probs],
    }


class Teach(BaseModel):
    cat_id: str
    clip: str
    label: str


@app.post("/api/teach")
def teach(t: Teach):
    label = t.label.strip()[:40]
    if t.cat_id not in cats:
        raise HTTPException(404, "Pick or add a cat first.")
    if t.clip not in _clips:
        raise HTTPException(410, "That recording expired. Record the meow again.")
    if not label:
        raise HTTPException(400, "Say what the cat meant.")
    cats[t.cat_id]["samples"].append({"label": label, "emb": _clips[t.clip].tolist()})
    save()
    return public(t.cat_id, cats[t.cat_id])


# Demo clips: clearly-cat (AST cat score > 0.5) and correctly classified. They're training clips, so they show the UI, not accuracy.
SAMPLES = {"hungry": "F_BLE01_EU_FN_DEL01_102.wav", "brushed": "B_CAN01_EU_FN_GIA01_309.wav", "alone": "I_BRA01_MC_MN_SIM01_301.wav"}


@app.get("/samples/{name}")
def sample(name: str):
    if name not in SAMPLES:
        raise HTTPException(404)
    return FileResponse(Path("data/dataset") / SAMPLES[name], media_type="audio/wav")


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)  # phones need https for the mic anyway: use a tunnel
