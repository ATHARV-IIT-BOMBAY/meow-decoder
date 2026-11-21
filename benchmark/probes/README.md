# Why the situation model doesn't transfer (v0.4.3)

One-off experiments behind README's "What we tried". Run from the repo root after `benchmark/wild.py download` and
with CatMeows in `data/dataset`, in this order:

    python benchmark/probes/transfer_variants.py    # embed real meows: band-limit, cut to the meow, level
    python benchmark/probes/transfer_score.py       # score those, and re-centring on real-recording statistics
    python benchmark/probes/transfer_farfield.py    # embed CatMeows through simulated rooms
    python benchmark/probes/transfer_far_score.py   # train with those copies
    python benchmark/probes/transfer_ood.py         # can unfamiliar recordings be flagged?
    python benchmark/probes/transfer_last.py        # train on real meows (leave-one-uploader-out); teaching under a room change
