"""Fast tests: no model downloads. Audio models are replaced by a fake embedder."""

import json
import shutil
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from meow_decoder import cli, server
from meow_decoder.bench import SPLITS, band_limit, benchmark, normalize, segments, teaching_curve
from meow_decoder.core import SR, Decoder, Head, decode

rng = np.random.default_rng(0)
QUIET = {"Cat": 0.9, "Meow": 0.8, "Purr": 0.0, "Caterwaul": 0.1, "Hiss": 0.0, "Speech": 0.0}
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def toy_data(n=60, d=8):
    y = np.array(["F", "B", "I"] * (n // 3))
    X = rng.normal(size=(n, d)).astype(np.float32) + (y[:, None] == np.array(["F", "B", "I"])).repeat(d // 3 + 1, 1)[:, :d] * 2
    return X, y


def fake_decoder(X, y, sounds=QUIET):
    d = Decoder.__new__(Decoder)  # skip weight download
    d.encoder, d.head = "beats", Head.fit(X, y)
    d.embed = lambda wave: (X[0], dict(sounds))
    return d


# ---------- model ----------


def test_head_roundtrip_and_probabilities(tmp_path):
    X, y = toy_data()
    head = Head.fit(X, y)
    head.cats, head.train_only = np.array(["a", "b"] * 30), np.zeros(60, bool)
    p = head.proba(X)
    assert np.allclose(p.sum(1), 1) and (head.classes[p.argmax(1)] == y).mean() > 0.8
    head.save(tmp_path / "beats.npz")
    again = Head.load(tmp_path / "beats.npz")
    assert np.allclose(again.proba(X), p) and again.encoder == "beats" and list(again.cats[:2]) == ["a", "b"]


def test_personalization_pulls_toward_taught_label():
    X, y = toy_data()
    head = Head.fit(X, y)
    meow = X[y == "F"].mean(0)
    taught = head.personalized([(meow, "B")] * 3)
    assert taught.proba(meow)[0][list(taught.classes).index("B")] > head.proba(meow)[0][list(head.classes).index("B")]
    assert head.personalized([]) is head


def test_signals_presets_and_not_a_cat():
    X, y = toy_data()
    d = fake_decoder(X, y)
    assert d.rank(X[0], QUIET)["signals"] == []
    purring = d.rank(X[0], {**QUIET, "Purr": 0.87})
    assert [s["title"] for s in purring["signals"]] == ["Purring"] and purring["is_cat"] and purring["has_meow"]
    assert not d.rank(X[0], {**QUIET, "Meow": 0.0, "Purr": 0.87})["has_meow"]
    noise = d.rank(X[0], {k: 0.0 for k in QUIET} | {"Speech": 0.9})  # talking is never a cat
    assert not noise["is_cat"] and not noise["has_meow"]
    top = d.rank(X[0], QUIET, d.head.personalized([(X[0], "out")] * 3))["results"][0]
    assert (top["key"], top["title"]) == ("out", "Wants to go out")


# ---------- audio ----------


def test_segments_find_the_meow():
    t = np.arange(SR) / SR
    meow = (0.5 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    room = (rng.normal(0, 0.002, SR * 8)).astype(np.float32)
    room[SR * 3 : SR * 4] += meow
    (a, b), *_ = segments(room)
    assert 2.8 * SR <= a <= 3.0 * SR and 4.0 * SR <= b <= 4.2 * SR
    assert segments(np.zeros(SR * 2, np.float32)) == [(0, SR * 2)]  # nothing stands out: whole clip


def test_band_limit_and_normalize():
    t = np.arange(SR) / SR
    tone = lambda f: np.sin(2 * np.pi * f * t).astype(np.float32)
    assert np.sqrt((band_limit(tone(6000)) ** 2).mean()) < 0.01 and np.sqrt((band_limit(tone(2000)) ** 2).mean()) > 0.6
    assert abs(np.sqrt((normalize(tone(500) * 0.001) ** 2).mean()) - 0.05) < 1e-3


@needs_ffmpeg
def test_decode_reads_audio_and_refuses_playlists(tmp_path):
    wav = tmp_path / "beep.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=0.5", str(wav)], check=True)
    assert abs(len(decode(wav.read_bytes())) - SR // 2) < SR // 50
    playlist = f"#EXTM3U\n#EXTINF:1,\nfile://{wav}\n#EXT-X-ENDLIST\n".encode()
    with pytest.raises(ValueError):
        decode(playlist)  # an uploaded playlist must not make ffmpeg read files on the server
    with pytest.raises(ValueError):
        decode(b"not audio at all")


# ---------- benchmark ----------


def test_benchmark_splits_are_whole_cats_with_interval():
    folds = json.loads(SPLITS.read_text())
    assert len(folds) == 21 and set(folds.values()) == {0, 1, 2, 3, 4}
    X, y = toy_data(n=63)
    cats = np.array(list(folds))[np.arange(63) % 21]
    r = benchmark(X, y, cats)
    assert 0 <= r["top1_ci"][0] <= r["top1"] <= r["top1_ci"][1] <= 1 and r["top2"] >= r["top1"] and r["cats"] == 21
    assert 0 < r["macro_recall"] <= 1 and set(r["recall"]) == {"B", "F", "I"}
    # augmented copies (train_only) help train but are never scored
    extra = benchmark(np.vstack([X, X + 0.01]), np.r_[y, y], np.r_[cats, cats], np.r_[np.zeros(63, bool), np.ones(63, bool)])
    assert extra["clips"] == 63 and extra["confidence"] > 0.4


def test_teaching_is_tested_on_other_sessions():
    folds = json.loads(SPLITS.read_text())
    X, y = toy_data(n=126)
    cats = np.array(list(folds))[np.arange(126) % 21]
    sessions = np.where(np.arange(126) < 63, "1", "2")
    cross = teaching_curve(X, y, cats, sessions, ks=(0, 2))
    same = teaching_curve(X, y, cats, sessions, ks=(2,), same_session=True)
    assert cross[0]["tested"] == 126 and cross[1]["tested"] == 126  # every test meow is from the other session
    assert same[0]["tested"] < 126


# ---------- app ----------


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "decode", lambda data: np.zeros(16000, np.float32))
    monkeypatch.setenv("MEOW_HOME", str(tmp_path))

    def make(sounds=QUIET, **kw):
        X, y = toy_data()
        return TestClient(server.build_app(fake_decoder(X, y, sounds), tmp_path, **kw))

    return make


def predict(c, cat_id="", headers=None):
    return c.post("/api/predict", files={"audio": ("m.webm", b"x")}, data={"cat_id": cat_id}, headers=headers or {}).json()


def test_app_flow_teach_undo_relabel_and_rerank(make_client):
    c = make_client()
    assert c.get("/api/config").json()["contrib"] is True
    cat = c.post("/api/cats", data={"name": "  Biscuit\n"}).json()
    assert cat["name"] == "Biscuit"
    r = predict(c, cat["id"])
    assert r["is_cat"] and len(r["results"]) == 3 and abs(sum(x["p"] for x in r["results"]) - 1) < 1e-3
    t = c.post("/api/teach", json={"cat_id": cat["id"], "clip": r["clip"], "label": "Wants to go out", "hour": 19}).json()
    assert t["counts"] == {"out": 1} and t["samples"][0]["hour"] == 19  # a preset's title maps to its key
    again = c.post("/api/teach", json={"cat_id": cat["id"], "clip": r["clip"], "label": "F"}).json()
    assert again["counts"] == {"F": 1}  # the same recording is relabelled, never counted twice
    assert predict(c, cat["id"])["taught"] == 1
    assert c.post("/api/rerank", json={"clip": r["clip"], "cat_id": ""}).json()["taught"] == 0
    undone = c.delete(f"/api/cats/{cat['id']}/samples/{again['sample_id']}").json()
    assert undone["counts"] == {} and c.delete(f"/api/cats/{cat['id']}/samples/{again['sample_id']}").status_code == 404


def test_sharing_keeps_good_meows_with_metadata_and_holds_back_custom_words(make_client, tmp_path):
    c = make_client()
    cat = c.post("/api/cats", data={"name": "Biscuit"}).json()
    r = predict(c, cat["id"])
    t = c.post(
        "/api/teach",
        json={"cat_id": cat["id"], "clip": r["clip"], "label": "B", "share": True, "session": "visit1", "hour": 7, "where": "home"},
        headers={"User-Agent": "Mozilla/5.0 (iPhone)"},
    ).json()
    assert t["shared"] and t["contribution_id"]
    r2 = predict(c, cat["id"])
    c.post("/api/teach", json={"cat_id": cat["id"], "clip": r2["clip"], "label": "Biscuit wants Priya", "share": True})
    rows = [json.loads(line) for line in (tmp_path / "contrib" / "metadata.jsonl").read_text().splitlines()]
    assert rows[0]["label"] == "B" and rows[0]["device"] == "phone" and rows[0]["where"] == "home" and rows[0]["hour"] == 7
    assert rows[1]["label"] == "custom" and rows[1]["needs_review"] and "Biscuit" not in json.dumps(rows[0])
    cli.main(["export-dataset", str(tmp_path / "out")])
    exported = (tmp_path / "out" / "metadata.csv").read_text()
    assert exported.count("\n") == 3 and "priya" not in exported and (tmp_path / "out" / "README.md").exists()


def test_sharing_refuses_talking(make_client):
    c = make_client(sounds={**QUIET, "Speech": 0.9})
    cat = c.post("/api/cats", data={"name": "Biscuit"}).json()
    r = predict(c, cat["id"])
    t = c.post("/api/teach", json={"cat_id": cat["id"], "clip": r["clip"], "label": "F", "share": True}).json()
    assert not t["shared"] and "talking" in t["share_note"] and t["counts"] == {"F": 1}


def test_visitors_only_see_their_own_cats(make_client):
    c = make_client()
    me, other = {"X-Meow-Owner": "me"}, {"X-Meow-Owner": "other"}
    cat = c.post("/api/cats", data={"name": "Biscuit"}, headers=me).json()
    assert [x["name"] for x in c.get("/api/cats", headers=me).json()] == ["Biscuit"]
    assert c.get("/api/cats", headers=other).json() == []
    r = predict(c, cat["id"], other)
    assert r["taught"] == 0
    assert c.post("/api/teach", json={"cat_id": cat["id"], "clip": r["clip"], "label": "B"}, headers=other).status_code == 404
    assert c.delete(f"/api/cats/{cat['id']}", headers=other).status_code == 404


def test_public_server_needs_owner_and_limits(make_client, monkeypatch):
    c = make_client(require_owner=True)
    assert c.get("/api/cats").status_code == 400
    h = {"X-Meow-Owner": "me"}
    monkeypatch.setattr(server, "MAX_CATS", 2)
    c = make_client(require_owner=True)
    for i in range(2):
        assert c.post("/api/cats", data={"name": f"c{i}"}, headers=h).status_code == 200
    assert c.post("/api/cats", data={"name": "c3"}, headers=h).status_code == 409
    big = c.post("/api/predict", files={"audio": ("m", b"x")}, headers={**h, "Content-Length": str(50 * 2**20)})
    assert big.status_code == 413


def test_rate_limit(make_client, monkeypatch):
    monkeypatch.setattr(server, "PREDICTS_PER_MINUTE", 2)
    c = make_client()
    assert [c.post("/api/predict", files={"audio": ("m", b"x")}).status_code for _ in range(3)] == [200, 200, 429]


def test_app_rejects_bad_requests(make_client):
    c = make_client()
    assert c.post("/api/teach", json={"cat_id": "nope", "clip": "x", "label": "B"}).status_code == 404
    assert c.post("/api/cats", data={"name": "  "}).status_code == 400
    assert c.post("/api/cats", data={"name": "x", "photo": "javascript:alert(1)"}).status_code == 400
    cat = c.post("/api/cats", data={"name": "Biscuit"}).json()
    r = predict(c, cat["id"])
    t = c.post("/api/teach", json={"cat_id": cat["id"], "clip": r["clip"], "label": "a\nb\x00  C"}).json()
    assert t["counts"] == {"a b c": 1}


def test_predict_keeps_going_past_bad_files(monkeypatch, tmp_path, capsys):
    X, y = toy_data()
    d = fake_decoder(X, y)
    d.predict = lambda f: (_ for _ in ()).throw(OSError("missing")) if "bad" in str(f) else d.rank(X[0], QUIET)
    monkeypatch.setattr("meow_decoder.core.Decoder", lambda enc: d)
    with pytest.raises(SystemExit):
        cli.main(["predict", "bad.wav", "good.wav"])
    out = capsys.readouterr()
    assert "good.wav" in out.out and "bad.wav" in out.err


def test_contribute_opens_a_pull_request(make_client, tmp_path, monkeypatch):
    c = make_client()
    cat = c.post("/api/cats", data={"name": "Biscuit"}).json()
    r = predict(c, cat["id"])
    c.post("/api/teach", json={"cat_id": cat["id"], "clip": r["clip"], "label": "F", "share": True})
    calls = []

    class FakeApi:
        def upload_folder(self, **kw):
            from pathlib import Path

            calls.append((kw, sorted(p.name for p in Path(kw["folder_path"]).iterdir())))
            return type("Info", (), {"pr_url": "https://hf.co/pr/1", "commit_url": ""})()

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    cli.main(["contribute"])
    kw, names = calls[0]
    assert kw["create_pr"] and kw["repo_type"] == "dataset" and kw["repo_id"] == cli.DATASET_REPO
    assert "metadata.csv" in names and any(n.endswith(".wav") for n in names) and "README.md" not in names


def test_score_someone_elses_predictions(tmp_path):
    folds = json.loads(SPLITS.read_text())
    names = [f"{'FBI'[i % 3]}_{cat}_EU_FN_X_1{i:02d}.wav" for i, cat in enumerate(list(folds) * 21) if i < 440]
    perfect = tmp_path / "p.csv"
    perfect.write_text("file_name,F,B,I\n" + "".join(f"{n},{int(n[0] == 'F')},{int(n[0] == 'B')},{int(n[0] == 'I')}\n" for n in names))
    r = cli.score_predictions(perfect)
    assert r["top1"] == 1.0 and r["clips"] == 440 and r["cats"] == 21
