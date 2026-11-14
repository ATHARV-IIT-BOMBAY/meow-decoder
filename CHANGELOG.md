# Changelog

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
