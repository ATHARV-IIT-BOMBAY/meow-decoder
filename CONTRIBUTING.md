# Contributing

Thanks for helping cats be understood a little better. In order of impact:

## 1. Share meows

More cats beat any model change. Use the app's **Also add this meow to the open cat dataset** box, then `meow export-dataset` and send us the bundle (an issue with a Hugging Face link works). Recordings of street cats, other breeds and other countries help most. Label what was really happening, not what you think the meow meant.

## 2. Beat the benchmark

Train anything you like, then score it on the fixed unseen-cat folds in `meow_decoder/splits.json` and [open a benchmark issue](.github/ISSUE_TEMPLATE/benchmark-result.md). Rules:

- No cat may appear in both training and test. Split by cat, never by clip.
- Augment after splitting, never before.
- Report first-guess accuracy with its interval, top-two accuracy, macro recall, log-loss and calibration error.
  The easiest way: save your out-of-fold probabilities as a CSV (`file_name,F,B,I`) and run
  `meow benchmark --predictions your.csv`, which computes all of them.
- Also report `meow benchmark --robust` if your model runs on audio: clean files flatter every model.
- Say if you used data beyond CatMeows.

## 3. Code

```bash
pip install -e ".[dev]"
pytest                    # fast, no model downloads
ruff check . && ruff format --check .
```

- Any change that touches predictions needs `meow benchmark` (and `--robust`) numbers before and after in the PR.
- Teaching results must be tested on a different recording session than the taught meows (`teaching_curve` does this).
- Keep the tests model-free so CI stays fast.
- Keep the app's copy honest: no "translation", no accuracy claims the benchmark doesn't back.

## When a new situation ships for everyone

A situation (one of the quick picks, or pain, fighting, mating calls, a mother calling kittens) becomes a global prediction once:

- shared meows with that label come from at least 10 different cats, and
- held-out-cat accuracy for it beats always-guessing by a clear margin, with whole recordings never split between training and test.

Until then it works only for cats whose owners teach it. `benchmark/sound_types.py` shows what happens when this rule is skipped.

## Ideas we'd welcome
- On-device inference: ONNX export so audio never leaves the phone.
- Better handling of noisy, real-world recordings.
