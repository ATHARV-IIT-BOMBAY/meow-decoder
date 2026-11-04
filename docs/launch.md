# Launch drafts

Drafts for you to post yourself. Edit freely; every number matches `benchmark/report.json`.

---

## Blog post

### Cat translators claim 90%. On a cat they've never heard, it's closer to 66%.

I wanted an app that tells me what the cats in my hostel are meowing about. Commercial apps advertise around 90% accuracy, and so do several papers. So I built one, tested it honestly, and found that most of that 90% comes from how the test is done rather than from the model.

**The trick.** A model that has heard a cat before recognises *that cat*, not what the meow means. If clips from the same cat end up in both training and test, the score measures memory. Many papers go further: they take 50 recordings per emotion, make 20 altered copies of each (added noise, pitch shifts, filters), and only then split at random. Copies of the same recording sit on both sides.

I reproduced this on the 50 public clips of a 10-emotion cat dataset. Same model, same data:

- Augment, then split randomly: **99.8%**
- Keep every copy of a recording on one side: **84.4%**
- Chance: 10%

**The honest number.** On CatMeows (440 meows, 21 cats, three situations: waiting for food, being brushed, alone somewhere new), I hold out whole cats. The best model I found, a bioacoustic encoder from Earth Species Project with a simple classifier on top, gets **65.9%** on the first guess (50% if you always guess the most common situation) and **89.1%** in its top two. Its percentages are honest: when it says 89%, it's right 88% of the time. An independent open benchmark, MeowSense, lands at 60% with a similar setup.

**What actually helps.** Teaching it your own cat. Four labelled meows per situation take a new cat from 66% to 81%. Cats really do each have their own voice.

**What it can't do.** Translate. Detect pain. Read food meows well: it catches only 30% of them, because to the model they sound like attention meows.

It's open source, with a browser demo, a `pip install`, the benchmark and the re-test: https://github.com/ATHARV-IIT-BOMBAY/meow-decoder

The real bottleneck is data: every open cat model learns from the same 21 European pet cats. If you have a cat, the app can share labelled meows to an open CC BY dataset.

---

## Hacker News

**Title:** Show HN: Meow Decoder – honest cat meow classifier, tested on cats it has never heard

**Text:** I built this to understand the cats in my hostel. Commercial cat translators claim ~90%. Holding out whole cats, the best model I could build gets 66% on three situations (50% baseline) and 89% in its top two, with calibrated percentages. Most published 90% figures come from augmenting before a random split; I reproduce that effect (99.8% vs 84.4% on the same clips). Teaching it four meows of your own cat gets you to 81%. Open source, browser demo, pip install, fixed benchmark folds, and an opt-in way to share labelled meows to an open dataset.

---

## Reddit

**r/MachineLearning [P]:** Meow Decoder: a cat vocalization benchmark that holds out whole cats, plus a reproduction of augment-before-split leakage (99.8% → 84.4%)

**r/cats:** I made a free, open-source app that guesses what your cat's meow is about and learns your cat's voice. It's honest about how often it's right (about 2 in 3 for a new cat, 4 in 5 once you teach it). Would love meows from your cats to make it better for everyone.

---

## Note for MeowSense (as a GitHub issue on sbor3937/MeowSense)

**Title:** Comparable unseen-cat results with bioacoustic BEATs embeddings (0.66)

Hi! Thanks for MeowSense; your grouped-by-cat protocol is the same one I use in Meow Decoder (https://github.com/ATHARV-IIT-BOMBAY/meow-decoder), and our AST numbers agree (your 0.60 ± 0.10, my 0.62). Swapping AST for Earth Species Project's `esp_aves2_sl_beats_all` encoder with a logistic-regression probe gave 0.659 first-guess accuracy, 0.891 top-two and a calibration error of 2.2% on fixed folds over whole cats. My fold assignment is in `meow_decoder/splits.json` if you'd like to compare like for like. Would you be open to sharing one set of folds and a joint leaderboard?

---

## Workshop paper outline

Possible venues: the DCASE workshop, or an audio or animal-behaviour workshop at NeurIPS, ICLR or ICASSP. Check each year's call for papers.

1. **Problem.** Inferring the context of domestic cat vocalizations; why evaluating on unseen individuals matters.
2. **Leakage.** Augment-before-split and same-cat splits; the 99.8% / 84.4% reproduction; an audit of reported numbers.
3. **Benchmark.** CatMeows with fixed whole-cat folds; metrics: first guess, top two, log-loss, calibration error.
4. **Encoders.** MFCC, AST, three ESP BEATs variants; bioacoustic pretraining improves both accuracy and calibration.
5. **Personalization.** A few labelled meows per cat: 66% → 81%.
6. **Open data.** The consent-based collection tool and first contributed meows.
7. **Limits.** Three contexts, two breeds, 8 kHz audio.
