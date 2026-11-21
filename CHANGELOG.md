# Changelog

## 0.6.0 (2026-11-21)

- **The app runs in the browser, on GitHub Pages**: https://atharv-iit-bombay.github.io/meow-decoder/. No server:
  `web/engine.js` answers the app's API calls on the device. AudioSet's AST runs through transformers.js (Xenova's
  quantized ONNX copy, pinned), the pitch model and teaching run in `web/dsp.js`, cats are kept in the browser, and
  recordings never leave it. Teaching there learns from AudioSet scores plus pitch (59% -> 65% on CatMeows across
  sessions, against 65% -> 72% with BEATs).
- `tests/test_web.py` runs the JavaScript under Node against the Python code; `benchmark/export_web.py` builds
  `web/model.json`.
- Audio is decoded straight at 16 kHz in the browser: resampling through 44.1 kHz moved AudioSet's "Cat" score on an
  8 kHz clip from 0.48 to 0.12.
- Asset paths and the service worker are relative, so the app works under a sub-path.

## 0.5.2 (2026-11-21)

App:
- The bundled samples show the real answer after playing, and whether the guess was first, second or missed.
- Untaught guesses draw the loudest meow's pitch line with what it means ("rose 1.5 semitones, rising meows lean
  food"). The server returns it as `pitch_line`.
- A lightly taught cat no longer reads "tuned to Biscuit · 90%": until about 3 taught meows per situation it says
  how many it has learned from and to treat the percentages as a guess.
- Phones: upload and the samples come before the empty result card, so they're on the first screen.
- Desktop: no more empty gap in the left column after a result.
- The cat row fades at the edge when more cats are off-screen; the photo picker matches the rest of the design.

Head-to-head with MeowSense on the real-world test (`benchmark/compare.py`), and why Meow-Omni 1 can't be compared:
see README, "Compared with other projects".

## 0.5.1 (2026-11-21)

- AST mode (commercial use) gets a yowl detector on AudioSet's 527 scores, trained only on CC0 and CC BY audio
  (`meow_decoder/detectors_ast.npz`): real fights and mating calls caught 12 → 20 of 24, false alarms 16 → 20 of 142.
  Hiss and purr detectors trained the same way didn't beat AudioSet's scores on the real-world test, so AST mode
  keeps those.
- `benchmark/detectors.py` takes `--encoder`, filters training audio by licence for AST mode, and decides what ships
  by a fixed rule: catch more than AudioSet, with more extra catches than extra false alarms.
- `benchmark/wild.py evaluate --encoder ast` writes `wild_report_ast.json`.

## 0.5.0 (2026-11-21)

**Situation guess for cats nobody has taught yet now comes from the meow's pitch contour.** On 179 real recordings from
Wikimedia Commons and Freesound the old guess was right 14% of the time, because it had learned CatMeows' recording
setup. Following the Meowsic finding that food meows tend to rise and vet meows to fall, a three-number pitch model
(`meow_decoder/pitch.npz`, `benchmark/pitch.py`) gets 45% first guess and 66% top two on 55 real meows from 42
uploaders. It rests on few clearly labelled stressed and attention meows, and the pitch tracker was tuned on these
clips; a fresh check on 7 more FSD50K meows went 4 of 7 (old model 1 of 7). Taught cats use their own meows, as before.

**Hissing and yowling detectors trained on real recordings** (`meow_decoder/detectors.npz`, `benchmark/detectors.py`):
FSD50K cat clips and other sounds, 19 Freesound fight and mating recordings, and CatMeows, with no uploader from the
real-world test. Real hisses caught 5 → 9 of 13; fights and mating calls 12 → 22 of 24. Hiss false alarms on non-cat
sounds (fans, steam) 73 → 5 of 450. With the AST encoder, AudioSet's scores are still used (licences). Purring stays
with AudioSet: the trained purr detector wasn't clearly better.

- API: results carry `guess_from` ("pitch", "taught" or "embedding"); `Decoder.rank(emb, sounds, head, pitch)`;
  `pitch_features(wave)`, `pitch_track(wave)`.
- The real-world yowl score now counts mating calls, as the app's description of yowling does.
- Tried and not shipped (numbers in README): mixing the embedding and pitch models; counting a detected hiss or
  yowl as hearing a cat (found 2 more cats, but called 3 more people imitating cats a cat).
- Online research: FSD50K for detector training; "meow-10k" on Hugging Face not used (unclear origin of its audio).

## 0.4.3 (2026-11-21)

- The app now says under every untaught situation guess that it's a rough guess (right about 1 time in 7 on real
  home recordings) and asks you to teach it. `meow predict` prints the same note.
- Tried to make the situation model work on real recordings: low-pass, cutting to the meow, re-centring, training
  through simulated rooms, the AST encoder, and adding real meows to training. None is good enough to ship; numbers
  in the README, scripts in `benchmark/probes/`. Teaching under a simulated room change: 48% → 60%.
- Hissing and yowling thresholds left as they are: on real recordings their scores overlap ordinary meows.

## 0.4.2 (2026-11-21)

- The real-world test now has 179 clips: 123 more from Freesound (`wild.py search --source freesound`, free API key),
  labelled the same strict way from what uploaders wrote. Results name how many uploaders they rest on.
- **The situation model doesn't transfer to ordinary recordings.** On 58 meows from 45 uploaders its first guess was
  right 8 times (14%); it called 2 of 48 food-begging meows "food". README, app and model card now say so next to
  the 66% CatMeows score.
- Purring: caught 26 of 30. Hissing: 5 of 13, 1 false alarm in 166. Yowling: 10 of 18. People imitating cats: 3 of
  12 called a cat.

## 0.4.1 (2026-11-19)

- New real-world test: 56 openly licensed cat recordings from Wikimedia Commons, labelled from what their
  uploaders wrote (`benchmark/wild.py`, `benchmark/wild_labels.csv`). Purring caught 14 of 18 with no false alarms;
  people saying "meow" were never called a cat. The situation model leaned towards "stressed" (15 of 29 meows) and
  got 1 of 5 situation-labelled meows right: too few to measure, enough to show it doesn't transfer well yet.
- Fixed: hissing was never detected. AudioSet scores real hisses around 0.02, below the old 0.2 threshold. The new
  threshold (0.01, 0.2% false alarms on CatMeows) caught 3 of 4 real hisses with no false alarms.
- The accuracy report page shows the corrected v0.4.0 numbers.

## 0.4.0 (2026-11-14)

An independent review of 0.3.1 found claims that didn't hold up and a list of bugs. This release corrects the
claims first.

**Corrected numbers**
- Teaching a cat: 66% → 81% was measured on meows from the same recording session as the taught ones. On a
  different session it is 65% → 72% (3 per situation). README, app and model card updated.
- Every headline score now has a 95% interval from resampling cats (65.9%, 57.0–74.3%) and macro recall (58.1%).
- New: accuracy under real recording conditions (browser codec 61.8%, heavy noise 59.3%, meow inside an 8 s
  recording 61.4%), and a test of how much is the room: background only 60.9%, meow only 63.9%.

**Tested and not shipped**, with numbers in the README: cutting clips to the meow, band-limiting to 4 kHz,
level normalisation (each lowered accuracy), and training on noisy copies (+1 point, inside the noise, 4× slower
teaching).

**App**
- Says "It's between X and Y" when its top answer is under 50%, and asks which one it was.
- Undo a taught meow, or remove any from the cat's card. Teaching the same recording twice relabels it.
- Taught-only labels show "learning" instead of a percentage until taught 3 times.
- Switching cats re-ranks the last recording for the new cat.
- A cat's card points out time-of-day patterns once there are enough taught meows.
- Sharing: unticked for every meow, asks where it was recorded, gives a contribution ID, refuses clips with
  talking or no cat, and keeps typed labels out of exports unless you opt in. `meow contribute` sends shared
  meows to the new open dataset as a pull request.
- Installs properly on phones (PNG icons, service worker) and opens offline.
- Fixed: recording could run past 8 s when the tab was hidden; a double tap during the permission prompt
  started two recordings; "not a cat" still showed a confident quote; the "shared" toast was never seen.

**Server and safety**
- Predictions run off the request loop, one at a time, so one upload can't freeze the app for others.
- Uploads over 10 MB are refused from their header; 30 predictions a minute per address; up to 20 cats and
  200 taught meows per cat.
- `cats.json` is written atomically under a lock, with embeddings stored compactly; browser IDs stored hashed.
- Off 127.0.0.1 the server requires each browser's ID and `meow serve` warns there are no logins.
- ffmpeg reads uploads with file and network access off and a 20 s timeout; labels and names are cleaned.
- Purring, hissing and yowling are listened for across the whole clip, not just the first 10 s.
- All model downloads pinned to exact revisions.

**Project**
- `meow benchmark --from-weights` (seconds, no audio), `--teaching`, `--robust`, `--predictions` (score your
  own model). Weights now store cat IDs.
- `meow predict` carries on past unreadable files.
- Tests 8 → 18, plus 4 browser tests of the real page; CI on Python 3.11–3.13, ruff, and a weekly check of the
  real models.
- librosa is now optional (`[train]`); version read from one place; requirements.txt fixed; NOTICE credits
  the bundled samples; code of conduct, security policy, pull-request template; tags for past releases.

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
