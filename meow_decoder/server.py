"""Web app: record or upload a meow, see the likeliest situations, teach it your cats.

Your cats live in MEOW_HOME (default ~/.meow-decoder): names, photos and meow embeddings, never audio.
Audio is kept only for meows you explicitly choose to share (MEOW_HOME/contrib), for the open dataset.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core import ENCODERS, SR, Decoder, decode

MAX_UPLOAD = 10 * 2**20
CLIP_CACHE = 50   # recent recordings kept in memory so "teach" needn't re-upload
STATIC = Path(__file__).resolve().parent / "static"


class Teach(BaseModel):
    cat_id: str
    clip: str
    label: str
    share: bool = False


def build_app(decoder: Decoder, home: Path, contrib: bool = True) -> FastAPI:
    home.mkdir(parents=True, exist_ok=True)
    store = home / "cats.json"
    cats: dict = json.loads(store.read_text()) if store.exists() else {}
    heads: dict = {}   # cat_id -> (n_samples, personalized Head)
    clips: dict = {}   # token -> (embedding, wave)

    def save():
        store.write_text(json.dumps(cats))

    def mine(cid, owner):  # each browser sees only its own cats (matters on a shared demo)
        c = cats.get(cid)
        return c if c and c.get("owner", "local") == owner else None

    def samples(cid):  # only meows embedded with the encoder running now
        return [(np.asarray(s["emb"], np.float32), s["label"]) for s in cats.get(cid, {}).get("samples", []) if s.get("encoder", "beats") == decoder.encoder]

    def head_for(cid):
        s = samples(cid)
        if not s:
            return decoder.head
        if heads.get(cid, (None,))[0] != len(s):
            heads[cid] = (len(s), decoder.head.personalized(s))
        return heads[cid][1]

    def public(cid, c):
        counts = {}
        for _, label in samples(cid):
            counts[label] = counts.get(label, 0) + 1
        return {"id": cid, "name": c["name"], "photo": c.get("photo"), "counts": counts}

    app = FastAPI(title="Meow Decoder")

    @app.get("/api/config")
    def config():
        return {"encoder": decoder.encoder, "encoder_info": ENCODERS[decoder.encoder], "contrib": contrib}

    Owner = Header("local", alias="X-Meow-Owner", max_length=64)

    @app.get("/api/cats")
    def list_cats(owner: str = Owner):
        return [public(k, c) for k, c in cats.items() if mine(k, owner)]

    @app.post("/api/cats")
    def add_cat(name: str = Form(..., max_length=40), photo: str = Form("", max_length=300_000), owner: str = Owner):
        if not name.strip():
            raise HTTPException(400, "Give your cat a name.")
        if photo and not photo.startswith("data:image/"):
            raise HTTPException(400, "Photo must be an image.")
        cid = uuid.uuid4().hex[:8]
        cats[cid] = {"name": name.strip(), "photo": photo or None, "samples": [], "owner": owner}
        save()
        return public(cid, cats[cid])

    @app.delete("/api/cats/{cid}")
    def delete_cat(cid: str, owner: str = Owner):
        if not mine(cid, owner):
            raise HTTPException(404, "No such cat.")
        cats.pop(cid, None)
        heads.pop(cid, None)
        save()
        return {"ok": True}

    @app.post("/api/predict")
    async def predict(audio: UploadFile, cat_id: str = Form(""), owner: str = Owner):
        data = await audio.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "That file is over 10 MB. Trim it to a few seconds of meowing.")
        try:
            wave = decode(data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        emb, sounds = decoder.embed(wave)
        cat_id = cat_id if mine(cat_id, owner) else ""
        token = uuid.uuid4().hex
        clips[token] = (emb, wave)
        if len(clips) > CLIP_CACHE:
            clips.pop(next(iter(clips)))
        return {"clip": token, "taught": len(samples(cat_id)), **decoder.rank(emb, sounds, head_for(cat_id))}

    @app.post("/api/teach")
    def teach(t: Teach, owner: str = Owner):
        label = t.label.strip()[:40]
        if not mine(t.cat_id, owner):
            raise HTTPException(404, "Pick or add a cat first.")
        if t.clip not in clips:
            raise HTTPException(410, "That recording expired. Record the meow again.")
        if not label:
            raise HTTPException(400, "Say what the cat meant.")
        emb, wave = clips[t.clip]
        cats[t.cat_id]["samples"].append({"label": label, "emb": emb.tolist(), "encoder": decoder.encoder})
        save()
        if t.share and contrib:
            share(home, wave, label, t.cat_id)
        return public(t.cat_id, cats[t.cat_id])

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def share(home: Path, wave: np.ndarray, label: str, cat_id: str):
    """Keep a consented meow for the open dataset: 16 kHz wav + one metadata line. The cat's name is never stored."""
    import soundfile as sf
    out = home / "contrib"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.wav"
    sf.write(out / name, wave, SR, subtype="PCM_16")
    row = {"file_name": name, "label": label, "cat_id": hashlib.sha256(cat_id.encode()).hexdigest()[:12],
           "submitted": datetime.date.today().isoformat(), "license": "CC-BY-4.0"}
    with open(out / "metadata.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def default_home() -> Path:
    return Path(os.environ.get("MEOW_HOME", Path.home() / ".meow-decoder"))
