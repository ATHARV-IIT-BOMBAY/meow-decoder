# Contributing

Thanks for helping cats be understood a little better. In order of impact:

## 1. Share meows

More cats beat any model change. Use the app's **Also add this meow to the open cat dataset** box, then `meow export-dataset` and send us the bundle (an issue with a Hugging Face link works). Recordings of street cats, other breeds and other countries help most. Label what was really happening, not what you think the meow meant.

## 2. Beat the benchmark

Train anything you like, then score it on the fixed unseen-cat folds in `meow_decoder/splits.json` and [open a benchmark issue](.github/ISSUE_TEMPLATE/benchmark-result.md). Rules:

- No cat may appear in both training and test. Split by cat, never by clip.
- Augment after splitting, never before.
- Report first-guess accuracy, top-two accuracy, log-loss and calibration error (`meow_decoder.core.benchmark` computes all four).
- Say if you used data beyond CatMeows.

## 3. Code

```bash
pip install -e ".[dev]"
pytest            # fast, no model downloads
```

- Any change that touches predictions needs `meow benchmark` numbers before and after in the PR.
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
