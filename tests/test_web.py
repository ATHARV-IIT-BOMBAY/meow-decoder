"""The browser version (web/dsp.js) must give the same answers as the Python code. Runs the JavaScript under Node."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from meow_decoder.core import SR, Head, pitch_features, pitch_line, segments

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="needs Node.js")
rng = np.random.default_rng(3)


def run_js(code: str, data) -> dict:
    script = f"import * as d from {json.dumps((ROOT / 'web' / 'dsp.js').as_uri())};\n" + code
    out = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(data), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def waves():
    t = np.arange(int(0.8 * SR)) / SR

    def glide(a, b):  # noqa: E306
        tone = 0.3 * np.sin(2 * np.pi * np.cumsum(np.linspace(a, b, len(t))) / SR)
        return (
            np.r_[np.zeros(SR // 2), tone, np.zeros(SR // 3), tone[::-1] * 0.5, np.zeros(SR // 2)]
            + rng.normal(0, 1e-3, len(t) * 2 + SR + SR // 3)
        ).astype(np.float32)

    return {"rising": glide(500, 800), "falling": glide(800, 450), "noise": rng.normal(0, 0.05, SR).astype(np.float32)}


def test_pitch_and_segments_match_python():
    ws = waves()
    js = run_js(
        """let s = ""; process.stdin.on("data", (c) => s += c).on("end", () => {
  const out = {};
  for (const [k, w] of Object.entries(JSON.parse(s))) { const a = Float32Array.from(w);
    out[k] = { seg: d.segments(a), feat: d.pitchFeatures(a).map((v) => Number.isNaN(v) ? null : v), line: d.pitchLine(a) }; }
  console.log(JSON.stringify(out)); });""",
        {k: w.tolist() for k, w in ws.items()},
    )
    for k, w in ws.items():
        assert js[k]["seg"] == [list(map(int, s)) for s in segments(w)], k
        py = pitch_features(w)
        assert np.allclose(np.array(js[k]["feat"], dtype=float), py, atol=1e-3, equal_nan=True), (k, js[k]["feat"], py)
        line = pitch_line(w)
        if line is None:
            assert js[k]["line"] is None
        else:
            assert abs(js[k]["line"]["rise"] - line["rise"]) < 0.02 and np.allclose(
                js[k]["line"]["semitones"], line["semitones"], atol=0.02
            )


def test_teaching_fit_matches_sklearn():
    m = json.loads((ROOT / "web" / "model.json").read_text())
    pick = rng.choice(len(m["teach"]["y"]), 150, replace=False).tolist()
    w = [20.0 if i < 5 else 1.0 for i in range(150)]  # the first five play the taught meows
    js = run_js(
        """let s = ""; process.stdin.on("data", (c) => s += c).on("end", () => {
  const { model, pick, w } = JSON.parse(s), X = d.unpackColumns(model.teach.X), rows = pick.map((i) => X[i]);
  const h = d.fitHead(rows, pick.map((i) => model.teach.y[i]), w, model.teach.C);
  console.log(JSON.stringify({ classes: h.classes, X: rows.map((r) => Array.from(r)), P: rows.map((r) => d.headProba(h, r)) }));
});""",
        {"model": m, "pick": pick, "w": w},
    )
    h = Head.fit(np.array(js["X"]), np.array(m["teach"]["y"])[pick], np.array(w))
    assert js["classes"] == h.classes.tolist()
    gap = np.abs(np.array(js["P"]) - h.proba(np.array(js["X"]))).max()
    assert gap < 0.02, gap
