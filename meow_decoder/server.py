"""Web app: record or upload a meow, see the likeliest situations, teach it your cats.

Your cats live in MEOW_HOME (default ~/.meow-decoder): names, photos and meow embeddings, never audio.
Audio is kept only for meows you explicitly choose to share (MEOW_HOME/contrib), for the open dataset.
"""

from __future__ import annotations

import base64
import collections
import datetime
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .core import CONTEXTS, ENCODERS, PRESETS, SR, Decoder, decode

MAX_UPLOAD = 10 * 2**20
CLIP_CACHE = 50  # recent recordings kept in memory so "teach" needn't re-upload
MAX_CATS = 20  # per owner
MAX_SAMPLES = 200  # taught meows per cat
PREDICTS_PER_MINUTE = 30  # per client address; each prediction costs about a second of CPU
SPEECH_MAX = 0.5  # shared meows with more speech than this are kept out of the open dataset
STATIC = Path(__file__).resolve().parent / "static"
WHERE = {"home", "outdoors", "vet", "car", "other"}


class Teach(BaseModel):
    cat_id: str
    clip: str
    label: str = Field(max_length=200)
    share: bool = False
    session: str = Field("", max_length=64)  # one random id per app visit, so a dataset can be split by session
    hour: int | None = Field(None, ge=0, le=23)
    where: str = Field("", max_length=16)


class Rerank(BaseModel):
    cat_id: str = ""
    clip: str


def clean_label(text: str) -> str:
    """Collapse whitespace, drop control characters, cap at 40; a preset's title or key maps to the key."""
    t = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", text)).strip()[:40]
    for key, (_, title, _) in {**CONTEXTS, **PRESETS}.items():
        if t.lower() in (key.lower(), title.lower()):
            return key
    return t.lower()


def owner_hash(owner: str) -> str:
    return hashlib.sha256(owner.encode()).hexdigest()[:32]


def pack(emb: np.ndarray) -> str:  # float32 embedding -> base64: ~4 KB instead of ~15 KB of JSON numbers
    return base64.b64encode(np.asarray(emb, np.float32).tobytes()).decode()


def unpack(v) -> np.ndarray:
    return np.frombuffer(base64.b64decode(v), np.float32) if isinstance(v, str) else np.asarray(v, np.float32)


def build_app(decoder: Decoder, home: Path, contrib: bool = True, require_owner: bool = False) -> FastAPI:
    """require_owner: refuse API calls without an X-Meow-Owner header (use whenever others can reach the server)."""
    home.mkdir(parents=True, exist_ok=True)
    store = home / "cats.json"
    cats: dict = json.loads(store.read_text()) if store.exists() else {}
    heads: dict = {}  # cat_id -> (samples signature, personalized Head)
    clips: dict = {}  # token -> {emb, sounds, wave}
    state_lock = threading.Lock()  # cats, clips and the file
    model_lock = threading.Lock()  # the audio models and refits run one at a time
    hits: dict = collections.defaultdict(collections.deque)

    def save():
        tmp = store.with_suffix(".tmp")
        tmp.write_text(json.dumps(cats))
        os.replace(tmp, store)  # atomic: a crash never leaves a half-written file

    def owner_of(header: str | None) -> str:
        if not header:
            if require_owner:
                raise HTTPException(400, "Missing X-Meow-Owner header.")
            header = "local"
        return owner_hash(header)

    def mine(cid, owner):  # each browser sees only its own cats
        c = cats.get(cid)
        if not c:
            return None
        stored = c.get("owner", owner_hash("local"))
        if stored == owner:
            return c
        if owner_hash(stored) == owner:  # cats saved before owner ids were hashed
            c["owner"] = owner
            return c
        return None

    def samples(cid):  # only meows embedded with the encoder running now
        return [s for s in cats.get(cid, {}).get("samples", []) if s.get("encoder", "beats") == decoder.encoder]

    def head_for(cid):
        s = samples(cid)
        if not s:
            return decoder.head
        sig = tuple((x.get("id"), x["label"]) for x in s)
        if heads.get(cid, (None,))[0] != sig:
            heads[cid] = (sig, decoder.head.personalized([(unpack(x["emb"]), x["label"]) for x in s]))
        return heads[cid][1]

    def public(cid, c):
        counts = collections.Counter(s["label"] for s in samples(cid))
        recent = [{"id": s.get("id"), "label": s["label"], "at": s.get("at"), "hour": s.get("hour")} for s in reversed(samples(cid))][:50]
        return {"id": cid, "name": c["name"], "photo": c.get("photo"), "counts": dict(counts), "samples": recent}

    def rank(entry, cid):
        with model_lock:
            return decoder.rank(entry["emb"], entry["sounds"], head_for(cid))

    app = FastAPI(title="Meow Decoder")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path == "/api/predict":
            if int(request.headers.get("content-length") or 0) > MAX_UPLOAD + 2**16:  # before reading the body
                return JSONResponse({"detail": "That file is over 10 MB. Trim it to a few seconds of meowing."}, 413)
            ip, now = request.client.host if request.client else "?", time.monotonic()
            q = hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= PREDICTS_PER_MINUTE:
                return JSONResponse({"detail": "Too many meows at once. Wait a minute and try again."}, 429)
            q.append(now)
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")  # after an upgrade, browsers must not keep the old page
        return response

    @app.get("/api/config")
    def config():
        return {"encoder": decoder.encoder, "encoder_info": ENCODERS[decoder.encoder], "contrib": contrib}

    @app.get("/api/cats")
    def list_cats(x_meow_owner: str | None = Header(None, max_length=64)):
        owner = owner_of(x_meow_owner)
        with state_lock:
            return [public(k, c) for k, c in list(cats.items()) if mine(k, owner)]

    @app.post("/api/cats")
    def add_cat(
        name: str = Form(..., max_length=40),
        photo: str = Form("", max_length=300_000),
        x_meow_owner: str | None = Header(None, max_length=64),
    ):
        owner = owner_of(x_meow_owner)
        name = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", name)).strip()
        if not name:
            raise HTTPException(400, "Give your cat a name.")
        if photo and not photo.startswith("data:image/"):
            raise HTTPException(400, "Photo must be an image.")
        with state_lock:
            if sum(1 for k in cats if mine(k, owner)) >= MAX_CATS:
                raise HTTPException(409, f"You can add up to {MAX_CATS} cats. Remove one first.")
            cid = uuid.uuid4().hex[:8]
            cats[cid] = {"name": name, "photo": photo or None, "samples": [], "owner": owner}
            save()
            return public(cid, cats[cid])

    @app.delete("/api/cats/{cid}")
    def delete_cat(cid: str, x_meow_owner: str | None = Header(None, max_length=64)):
        owner = owner_of(x_meow_owner)
        with state_lock:
            if not mine(cid, owner):
                raise HTTPException(404, "No such cat.")
            cats.pop(cid, None)
            heads.pop(cid, None)
            save()
        return {"ok": True}

    @app.delete("/api/cats/{cid}/samples/{sid}")
    def delete_sample(cid: str, sid: str, x_meow_owner: str | None = Header(None, max_length=64)):
        owner = owner_of(x_meow_owner)
        with state_lock:
            c = mine(cid, owner)
            if not c:
                raise HTTPException(404, "No such cat.")
            before = len(c["samples"])
            c["samples"] = [s for s in c["samples"] if s.get("id") != sid]
            if len(c["samples"]) == before:
                raise HTTPException(404, "That meow was already removed.")
            save()
            return public(cid, c)

    @app.post("/api/predict")
    async def predict(audio: UploadFile, cat_id: str = Form(""), x_meow_owner: str | None = Header(None, max_length=64)):
        owner = owner_of(x_meow_owner)
        data = await audio.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "That file is over 10 MB. Trim it to a few seconds of meowing.")
        try:
            wave = await run_in_threadpool(decode, data)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

        def work():
            with model_lock:
                emb, sounds = decoder.embed(wave)
            with state_lock:
                cid = cat_id if mine(cat_id, owner) else ""
                token = uuid.uuid4().hex
                clips[token] = {"emb": emb, "sounds": sounds, "wave": wave}
                while len(clips) > CLIP_CACHE:
                    clips.pop(next(iter(clips)))
                taught = len(samples(cid))
            return {"clip": token, "taught": taught, **rank(clips[token], cid)}

        return await run_in_threadpool(work)

    @app.post("/api/rerank")
    def rerank(r: Rerank, x_meow_owner: str | None = Header(None, max_length=64)):
        """Rank a recent recording again for another cat, without re-uploading it."""
        owner = owner_of(x_meow_owner)
        with state_lock:
            entry = clips.get(r.clip)
            if not entry:
                raise HTTPException(410, "That recording expired. Record the meow again.")
            cid = r.cat_id if mine(r.cat_id, owner) else ""
            taught = len(samples(cid))
        return {"clip": r.clip, "taught": taught, **rank(entry, cid)}

    @app.post("/api/teach")
    def teach(t: Teach, request: Request, x_meow_owner: str | None = Header(None, max_length=64)):
        owner = owner_of(x_meow_owner)
        label = clean_label(t.label)
        if not label:
            raise HTTPException(400, "Say what the cat meant.")
        with state_lock:
            c = mine(t.cat_id, owner)
            if not c:
                raise HTTPException(404, "Pick or add a cat first.")
            entry = clips.get(t.clip)
            if not entry:
                raise HTTPException(410, "That recording expired. Record the meow again.")
            existing = next((s for s in c["samples"] if s.get("clip") == t.clip), None)
            if existing:  # the same recording taught again: relabel it, never count it twice
                existing["label"] = label
                sample = existing
            else:
                if len(c["samples"]) >= MAX_SAMPLES:
                    raise HTTPException(409, f"{c['name']} already has {MAX_SAMPLES} taught meows. Remove some first.")
                sample = {
                    "id": uuid.uuid4().hex[:10],
                    "clip": t.clip,
                    "label": label,
                    "emb": pack(entry["emb"]),
                    "encoder": decoder.encoder,
                    "hour": t.hour,
                    "at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                }
                c["samples"].append(sample)
            save()
            out = public(t.cat_id, c)
        out["sample_id"] = sample["id"]
        if t.share and contrib and not existing:
            out.update(share(home, entry, label, t, request.headers.get("user-agent", "")))
        return out

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def share(home: Path, entry: dict, label: str, t: Teach, user_agent: str) -> dict:
    """Keep a consented meow for the open dataset: 16 kHz wav + one metadata line.

    The cat's name is never stored. Clips that don't sound like a cat or contain speech are refused.
    Free-text labels are kept apart (label "custom") so a reviewer can check them before publishing.
    """
    import soundfile as sf

    sounds = entry["sounds"]
    if max(v for k, v in sounds.items() if k != "Speech") < 0.05:
        return {"shared": False, "share_note": "Not shared: it doesn't sound enough like a cat."}
    if sounds.get("Speech", 0) > SPEECH_MAX:
        return {"shared": False, "share_note": "Not shared: there's talking in it. Record the cat on its own."}
    out = home / "contrib"
    out.mkdir(parents=True, exist_ok=True)
    cid = uuid.uuid4().hex[:12]
    sf.write(out / f"{cid}.wav", entry["wave"], SR, subtype="PCM_16")
    known = label in CONTEXTS or label in PRESETS
    row = {
        "file_name": f"{cid}.wav",
        "contribution_id": cid,
        "label": label if known else "custom",
        "cat_id": hashlib.sha256(t.cat_id.encode()).hexdigest()[:12],
        "session": hashlib.sha256(t.session.encode()).hexdigest()[:12] if t.session else "",
        "device": "phone" if re.search(r"Mobi|Android|iPhone", user_agent) else "computer",
        "hour": t.hour,
        "where": t.where if t.where in WHERE else "",
        "seconds": round(len(entry["wave"]) / SR, 2),
        "submitted": datetime.date.today().isoformat(),
        "license": "CC-BY-4.0",
    }
    if not known:
        row.update(label_text=label, needs_review=True)
    with open(out / "metadata.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return {"shared": True, "contribution_id": cid}


def default_home() -> Path:
    return Path(os.environ.get("MEOW_HOME", Path.home() / ".meow-decoder"))
