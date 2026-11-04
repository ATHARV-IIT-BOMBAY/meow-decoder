# Changelog

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
