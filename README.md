# Meow Decoder

Record a meow and get the three likeliest reasons behind it, with honest percentages. Teach it your own cat and it gets better at that cat's voice.

It doesn't translate. Cats don't meow in sentences. It guesses which situation a meow sounds like, and tells you how sure it is.

## How well it works

All numbers are measured on cats the model never trained on (5-fold cross-validation grouped by cat), so they reflect meeting a new cat. Reproduce them with `python evaluate.py`, which writes `report.json`.

| | Accuracy |
|---|---|
| Always guess the most common situation | 50% |
| Classic MFCC features | 57.5% |
| AST general-audio embeddings | 62% |
| **Bioacoustic BEATs embeddings (shipped)** | **66%** |
| Right answer in the top two | 89% |
| After teaching 4 meows per situation from that cat | 81% |
| Cats it has already heard | 74% |

- **Its percentages are honest.** When it says about 89%, it is right 88% of the time (expected calibration error 2.2%, down from 4.7% with AST).
- **Food meows are the hardest.** It catches 84% of "stressed somewhere new" meows, 60% of "enjoying attention" and only 30% of "food, please".
- **Every cat is different.** Accuracy on a single unseen cat ranges from 20% to 95%, which is why teaching your cat helps so much.

### Why published 90% figures don't compare

Many cat-sound papers augment their clips (noise, pitch shift, filtering) and then split them randomly, so altered copies of the same recording end up in both training and test. On the 50 public clips of the Cat Sound Classification V2 dataset (10 emotions), the same model scores:

| Split | Accuracy |
|---|---|
| Augment, then split randomly | 99.8% |
| Keep every copy of a clip on one side | 84% |
| Chance | 10% |

[MeowSense](https://github.com/sbor3937/MeowSense), the other open benchmark that tests on unseen cats, reports 60% ± 10 on the same data, in line with these results.

## Run it

Needs Python 3.12 and `ffmpeg`.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data && curl -L -o data/catmeows.zip "https://zenodo.org/records/4008297/files/dataset.zip?download=1" && unzip -q data/catmeows.zip -d data
.venv/bin/python meow.py      # embeds the meows, prints held-out accuracy, saves model.joblib + base.npz
.venv/bin/python app.py       # http://localhost:8000
```

For `evaluate.py`'s paper re-test, also download the Cat Sound V2 samples into `data/catsound/`:

```bash
mkdir -p data/catsound && curl -L -o data/catsound/s.zip "https://zenodo.org/api/records/4724180/files/CAT_SOUND_DB_SAMPLES.zip/content" && unzip -q data/catsound/s.zip -d data/catsound
```

Phones only allow the microphone over https. To try it on a phone, tunnel it:

```bash
cloudflared tunnel --url http://localhost:8000
```

Anyone with that link can add and remove cats, so share it only with people you trust.

## How it works

- `meow.py`: each clip goes through two frozen models. Earth Species Project's bioacoustic BEATs encoder (`esp_aves2_sl_beats_all`) gives the embedding a logistic regression classifies into the three situations. MIT's AudioSet AST checks whether it sounds like a cat at all and listens for purring, hissing and yowling.
- `app.py`: FastAPI server. For a cat you've taught, it refits the classifier on the base meows plus that cat's meows (weighted 20×). It stores embeddings only, never your audio, in `data/mine/cats.json`.
- `static/index.html`: the whole front end. The cat is the record button; its pupils follow the sound level and its face changes with the answer.
- `evaluate.py`: the full held-out evaluation: confusion matrix, calibration, per-cat accuracy, personalization curve and the paper re-test.

## Data, models and licenses

- Code: MIT (see `LICENSE`).
- [CatMeows](https://zenodo.org/records/4008297) (Ludovico et al., 2020): 440 meows from 21 cats in three situations. CC BY 4.0.
- [Cat Sound Classification Dataset V2](https://zenodo.org/records/4724180) (Pandeya et al.): 10 emotion classes; 50 public sample clips used for the re-test. CC BY 4.0.
- [esp-aves2-sl-beats-all](https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-beats-all) (Earth Species Project): **CC BY-NC-SA 4.0, non-commercial**. To use this commercially, switch `meow.py` back to AST embeddings (about 4 points less accurate).
- [AST fine-tuned on AudioSet](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (MIT).

## Limits

- Three situations only: food, attention, stressed somewhere unfamiliar. It can't detect pain or illness. If a cat seems unwell, see a vet.
- Trained on 21 Maine Coon and European Shorthair pet cats recorded by the University of Milan. Street cats and other breeds may sound different.
- The data is 8 kHz audio, so the models never hear above 4 kHz.
