---
license: other
license_name: mixed-see-card
library_name: meow-decoder
pipeline_tag: audio-classification
tags: [cats, bioacoustics, meow, audio-classification]
datasets: [CatMeows]
---

# Meow Decoder classifier heads

Small logistic-regression heads that turn a meow's audio embedding into the likeliest situation:
waiting for food, enjoying attention, or stressed somewhere new. Used by
[meow-decoder](https://github.com/ATHARV-IIT-BOMBAY/meow-decoder), which downloads them automatically.

| File | Encoder | First guess right (unseen cats) | Top two | Calibration error | License |
|---|---|---|---|---|---|
| `beats.npz` | Earth Species Project `esp_aves2_sl_beats_all` | 65.9% | 89.1% | 2.2% | CC BY-NC-SA 4.0 (follows the encoder) |
| `ast.npz` | MIT AST fine-tuned on AudioSet | 61.8% | 84.1% | 4.7% | CC BY 4.0 (follows the CatMeows data) |

Always guessing the most common situation scores 50.2%. Scores are 5-fold cross-validation over whole cats
(`splits.json` in the code repo), so every test meow comes from a cat the head never saw.

## Format

Plain NumPy arrays, no pickle: `classes`, `mean`, `scale`, `coef`, `intercept` (standardize, then multinomial
logistic regression), plus `X`, `y` (the 440 training embeddings, kept so the app can refit with your own cat's meows).

```python
from meow_decoder import Decoder
Decoder("beats").predict("meow.wav")["results"]
```

## Training data

[CatMeows](https://zenodo.org/records/4008297) (Ludovico et al., 2020): 440 meows from 21 Maine Coon and European
Shorthair cats, CC BY 4.0. Audio is 8 kHz, so nothing above 4 kHz was heard.

## Limits

Three situations only. It cannot detect pain or illness. Pet cats from one dataset; other cats may sound different.
