"""The real page in a real browser, against the real server with a fake audio model (no downloads).

Checks what people actually see, which the API tests can't: the v0.3.0 purr bug passed every API test.
Needs `pip install playwright && python -m playwright install chromium`; skipped otherwise.
"""

import socket
import threading
import time

import numpy as np
import pytest

pw = pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402

from meow_decoder import server  # noqa: E402
from meow_decoder.core import Decoder, Head  # noqa: E402

QUIET = {"Cat": 0.9, "Meow": 0.8, "Purr": 0.0, "Caterwaul": 0.1, "Hiss": 0.0, "Speech": 0.0}
SOUNDS = {  # which fake result each sample button produces, keyed by the uploaded file's length
    "purr": {**QUIET, "Meow": 0.0, "Purr": 0.9},
    "noise": dict.fromkeys(QUIET, 0.0),
}


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    rng = np.random.default_rng(0)
    y = np.array(["F", "B", "I"] * 20)
    X = rng.normal(size=(60, 8)).astype(np.float32) + (y[:, None] == np.array(["F", "B", "I"])).repeat(3, 1)[:, :8] * 2
    d = Decoder.__new__(Decoder)
    d.encoder, d.head, d.pitch_head = "beats", Head.fit(X, y), None  # UI logic only: guesses come from the fake embedding
    mode = {"next": "confident"}
    d.embed = lambda wave: (
        X[0] if mode["next"] != "unsure" else (X[0] + X[1] + X[2]) / 3,
        SOUNDS.get(mode["next"], QUIET),
    )
    real_decode, server.decode = server.decode, lambda data: np.zeros(16000, np.float32)
    app = server.build_app(d, tmp_path_factory.mktemp("home"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", mode
    srv.should_exit = True
    server.decode = real_decode


@pytest.fixture
def page(base_url):
    with pw.sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 390, "height": 844})
        pg.goto(base_url[0])
        yield pg, base_url[1]
        browser.close()


def play(pg, mode, sample, kind):
    mode["next"] = kind
    pg.click(f'[data-sample="{sample}"]')
    pg.wait_for_selector("#result:not([hidden])")
    pg.wait_for_function("!document.querySelector('#status strong').textContent.startsWith('Thinking')")


def test_purr_hides_the_meow_guess(page):
    pg, mode = page
    play(pg, mode, "purring.mp3", "purr")
    assert pg.is_visible(".signal") and "Purring" in pg.inner_text(".signal")
    assert not pg.is_visible("#opts") and not pg.is_visible("#bubble") and pg.is_visible("#nomeow")


def test_noise_is_not_a_cat(page):
    pg, mode = page
    play(pg, mode, "hungry.wav", "noise")
    assert pg.is_visible("#warn") and not pg.is_visible("#opts") and not pg.is_visible("#nomeow")


def test_unsure_says_its_between_two(page):
    pg, mode = page
    play(pg, mode, "hungry.wav", "unsure")
    assert "It's between" in pg.inner_text("#quote") and pg.is_visible("#opts")
    # a bundled sample shows the real answer and how the guess did
    truth, order = pg.inner_text("#truth"), [t.split("\n")[0] for t in pg.locator("#opts .t").all_inner_texts()]
    expected = ("✅", "🟡", "❌")[min(order.index("Food, please"), 2)]
    assert "waiting for food" in truth and expected in truth


def test_teach_then_undo(page):
    pg, mode = page
    pg.click("text=+ Add a cat")
    pg.fill("#catName", "Biscuit")
    pg.click("#addSave")
    play(pg, mode, "hungry.wav", "confident")
    assert pg.is_visible("#shareMeow") and not pg.is_checked("#shareMeow")
    pg.click("#teach >> text=🍗 Food")
    pg.wait_for_selector("#teach >> text=Undo")
    assert "Food 1/3" in pg.inner_text(".dots")
    pg.click("#teach >> text=Undo")
    pg.wait_for_function("document.querySelector('.dots').textContent.includes('Food 0/3')")
    assert pg.is_enabled("#teach >> text=🍗 Food")
