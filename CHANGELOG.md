# Changelog

## 0.3.1 (2026-11-04)

- Fixed: for a clip with no meow (a purr), the situation bars still showed under "no situation to guess".
- The server tells browsers to re-check the page, so an upgrade never leaves an old version on screen.
- README rewritten to cover everything: the app, teaching, the CLI, Python and HTTP APIs, settings and privacy, full accuracy breakdown, encoders, deployment, troubleshooting and roadmap.
- The demo GIF now ends with the purring screen.

## 0.3.0 (2026-11-04)

- Purring, hissing and yowling are now first-class results (`signals`), each with what it usually means. The cat's face reacts to them.
- Clips with no meow (a purr, a hiss) no longer get a meaningless situation guess (`has_meow`).
- Four quick-pick situations to teach your own cat: saying hello, wants to go out, wants to play, annoyed.
- A purring sample in the app and notebook.
- `benchmark/sound_types.py`: why pain, mating, fighting and mother-call detection aren't shipped (80% → 17% once whole recordings are held out) and how often each sound detector false-alarms.

## 0.2.0 (2026-11-04)

- Installable package with a `meow` command (`predict`, `serve`, `train`, `benchmark`, `export-dataset`) and a Python API (`Decoder`).
- Default encoder is now Earth Species Project's bioacoustic BEATs: first guess on unseen cats 61.8% → 65.9%, top two 84.1% → 89.1%, calibration error 4.7% → 2.2%.
- `--encoder ast` keeps a commercial-safe mode.
- Weights published on Hugging Face as plain NumPy arrays (no pickle) and downloaded on first use.
- Fixed unseen-cat benchmark folds (`meow_decoder/splits.json`) and `meow benchmark`.
- Opt-in sharing of taught meows for an open CC BY 4.0 dataset, with `meow export-dataset`.
- Each browser only sees its own cats, so a shared server or demo keeps cats private.
- Models load at startup instead of on the first meow.
- Tests and CI.
- Free "Try it" Colab notebook: samples, upload, record in the browser, or open the full app. (Hugging Face Docker Spaces now need a paid plan; `hf/space/` is ready if that changes.)

## 0.1.0 (2026-11-04)

- First release: web app with the reacting cat, per-cat teaching, AST embeddings, held-out evaluation and the augment-then-split re-test.
