"""Fast tests: no model downloads. Audio models are replaced by a fake embedder."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from meow_decoder import cli, server
from meow_decoder.core import SPLITS, Decoder, Head, benchmark

rng = np.random.default_rng(0)


def toy_data(n=60, d=8):
    y = np.array(["F", "B", "I"] * (n // 3))
    X = rng.normal(size=(n, d)).astype(np.float32) + (y[:, None] == np.array(["F", "B", "I"])).repeat(d // 3 + 1, 1)[:, :d] * 2
    return X, y


def test_head_roundtrip_and_probabilities(tmp_path):
    X, y = toy_data()
    head = Head.fit(X, y)
    p = head.proba(X)
    assert np.allclose(p.sum(1), 1) and (head.classes[p.argmax(1)] == y).mean() > 0.8
    head.save(tmp_path / "beats.npz")
    again = Head.load(tmp_path / "beats.npz")
    assert np.allclose(again.proba(X), p) and again.encoder == "beats"


def test_personalization_pulls_toward_taught_label():
    X, y = toy_data()
    head = Head.fit(X, y)
    meow = X[y == "F"].mean(0)
    taught = head.personalized([(meow, "B")] * 3)
    assert taught.proba(meow)[0][list(taught.classes).index("B")] > head.proba(meow)[0][list(head.classes).index("B")]
    assert head.personalized([]) is head


def test_custom_label_becomes_a_new_option():
    X, y = toy_data()
    taught = Head.fit(X, y).personalized([(X[0], "wants to go out")] * 3)
    assert "wants to go out" in taught.classes


def test_signals_and_presets_in_results():
    X, y = toy_data()
    d = Decoder.__new__(Decoder)
    d.encoder, d.head = "beats", Head.fit(X, y)
    quiet = {"Cat": 0.9, "Meow": 0.8, "Purr": 0.0, "Caterwaul": 0.1, "Hiss": 0.0}
    assert d.rank(X[0], quiet)["signals"] == []
    purring = d.rank(X[0], {**quiet, "Purr": 0.87})
    assert [s["title"] for s in purring["signals"]] == ["Purring"] and purring["is_cat"] and purring["has_meow"]
    assert not d.rank(X[0], {**quiet, "Meow": 0.0, "Purr": 0.87})["has_meow"]
    taught = d.head.personalized([(X[0], "out")] * 3)
    top = d.rank(X[0], quiet, taught)["results"][0]
    assert (top["key"], top["title"]) == ("out", "Wants to go out")


def test_benchmark_splits_are_whole_cats():
    folds = json.loads(SPLITS.read_text())
    assert len(folds) == 21 and set(folds.values()) == {0, 1, 2, 3, 4}
    X, y = toy_data(n=63)
    cats = np.array(list(folds))[np.arange(63) % 21]
    r = benchmark(X, y, cats)
    assert 0 <= r["top1"] <= 1 and r["top2"] >= r["top1"] and r["cats"] == 21


@pytest.fixture
def client(tmp_path, monkeypatch):
    X, y = toy_data()
    d = Decoder.__new__(Decoder)  # skip weight download
    d.encoder, d.head = "beats", Head.fit(X, y)
    d.embed = lambda wave: (X[0], {"Cat": 0.9, "Meow": 0.8, "Purr": 0.0, "Caterwaul": 0.0, "Hiss": 0.0})
    monkeypatch.setattr(server, "decode", lambda data: np.zeros(16000, np.float32))
    monkeypatch.setenv("MEOW_HOME", str(tmp_path))
    return TestClient(server.build_app(d, tmp_path))


def test_app_flow_teach_share_and_export(client, tmp_path):
    assert client.get("/api/config").json()["contrib"] is True
    cat = client.post("/api/cats", data={"name": "Biscuit"}).json()
    r = client.post("/api/predict", files={"audio": ("m.webm", b"x")}, data={"cat_id": cat["id"]}).json()
    assert r["is_cat"] and len(r["results"]) == 3 and abs(sum(x["p"] for x in r["results"]) - 1) < 1e-3
    taught = client.post("/api/teach", json={"cat_id": cat["id"], "clip": r["clip"], "label": "B", "share": True}).json()
    assert taught["counts"] == {"B": 1}
    meta = [json.loads(l) for l in (tmp_path / "contrib" / "metadata.jsonl").read_text().splitlines()]
    assert meta[0]["label"] == "B" and "Biscuit" not in json.dumps(meta) and (tmp_path / "contrib" / meta[0]["file_name"]).exists()
    assert client.post("/api/predict", files={"audio": ("m.webm", b"x")}, data={"cat_id": cat["id"]}).json()["taught"] == 1
    cli.main(["export-dataset", str(tmp_path / "out")])
    assert (tmp_path / "out" / "metadata.csv").read_text().count("\n") == 2 and (tmp_path / "out" / "README.md").exists()


def test_visitors_only_see_their_own_cats(client):
    me, other = {"X-Meow-Owner": "me"}, {"X-Meow-Owner": "other"}
    cat = client.post("/api/cats", data={"name": "Biscuit"}, headers=me).json()
    assert [c["name"] for c in client.get("/api/cats", headers=me).json()] == ["Biscuit"]
    assert client.get("/api/cats", headers=other).json() == []
    r = client.post("/api/predict", files={"audio": ("m.webm", b"x")}, data={"cat_id": cat["id"]}, headers=other).json()
    assert client.post("/api/teach", json={"cat_id": cat["id"], "clip": r["clip"], "label": "B"}, headers=other).status_code == 404
    assert client.delete(f"/api/cats/{cat['id']}", headers=other).status_code == 404


def test_app_rejects_bad_requests(client):
    assert client.post("/api/teach", json={"cat_id": "nope", "clip": "x", "label": "B"}).status_code == 404
    assert client.post("/api/cats", data={"name": "  "}).status_code == 400
    assert client.post("/api/cats", data={"name": "x", "photo": "javascript:alert(1)"}).status_code == 400
