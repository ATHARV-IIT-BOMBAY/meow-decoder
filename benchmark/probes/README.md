# Why the situation model doesn't transfer (v0.4.3)

One-off experiments behind README's "What we tried". Run from the repo root after `benchmark/wild.py download` and
with CatMeows in `data/dataset`, in this order:

    python benchmark/probes/transfer_variants.py    # embed real meows: band-limit, cut to the meow, level
    python benchmark/probes/transfer_score.py       # score those, and re-centring on real-recording statistics
    python benchmark/probes/transfer_farfield.py    # embed CatMeows through simulated rooms
    python benchmark/probes/transfer_far_score.py   # train with those copies
    python benchmark/probes/transfer_ood.py         # can unfamiliar recordings be flagged?
    python benchmark/probes/transfer_last.py        # train on real meows (leave-one-uploader-out); teaching under a room change

# v0.5.0: pitch contour and trained detectors

    python benchmark/probes/pitch.py              # pitch features with librosa's pYIN (first test of the Meowsic cue)
    python benchmark/probes/pitch_score.py        # does contour shape separate food from stressed? CatMeows and real
    python benchmark/probes/pitch_yin.py          # the same with the shipped NumPy YIN tracker (core.pitch_features)
    python benchmark/probes/pitch_yin_score.py
    python benchmark/probes/pitch_combo.py        # mixing the embedding and pitch models (didn't help)
    python benchmark/probes/fresh_check.py        # 7 FSD50K meows labelled before scoring
    python benchmark/probes/teach_vs_pitch.py     # teaching vs the pitch guess, through simulated rooms
    python benchmark/probes/detectors_data.py     # first detector experiment (real clips in training folds)
    python benchmark/probes/detectors.py

The shipped models are built by benchmark/pitch.py and benchmark/detectors.py.
