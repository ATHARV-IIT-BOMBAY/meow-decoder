"""Which extra sound types can we honestly ship? Run from the repo root -> benchmark/sound_types.json

1. A 10-emotion classifier on the 50 public Cat Sound V2 clips, scored two ways: leaving out one clip,
   and leaving out a whole recording run (same source prefix, clip numbers within 3 of each other).
   The first measures recognising the recording; the second, recognising the emotion.
2. How often the AudioSet purr / hiss / yowl detectors fire on 440 ordinary CatMeows meows (false alarms)
   and on each Cat Sound V2 class.
"""
import json
import re
from pathlib import Path

import numpy as np

from meow_decoder.core import SIGNALS, Head, analyze, catmeows, load

root = Path("data/catsound/CAT_SOUND_DB_SAMPLES")
files = sorted(f for d in root.iterdir() if d.is_dir() for f in d.iterdir())
y = np.array([f.parent.name for f in files])
key = [re.match(r"(\D+?)(\d+)", f.name).groups() for f in files]
prefix, num = np.array([k[0] for k in key]), np.array([int(k[1]) for k in key])

order = np.lexsort((num, prefix, y))
run = np.zeros(len(y), int)
for k in range(1, len(order)):
    i, j = order[k], order[k - 1]
    run[i] = run[j] + (y[i] != y[j] or prefix[i] != prefix[j] or num[i] - num[j] > 3)

report = {"clips": len(y), "classes": int(len(set(y))), "runs_per_class": {c: int(len(set(run[y == c]))) for c in sorted(set(y))}}
for enc in ["beats", "ast"]:
    X = np.stack([analyze(load(f), enc)[0] for f in files])
    clip_hits = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        h = Head.fit(X[m], y[m])
        clip_hits.append(h.classes[h.proba(X[i]).argmax()] == y[i])
    run_hits, untestable = {}, []
    for r in np.unique(run):
        m = run != r
        c = y[~m][0]
        if c not in set(y[m]):  # the class's only recording: nothing left to learn it from
            untestable.append(c)
            continue
        h = Head.fit(X[m], y[m])
        run_hits.setdefault(c, []).extend((h.classes[h.proba(X[~m]).argmax(1)] == y[~m]).tolist())
    flat = [x for v in run_hits.values() for x in v]
    report[enc] = {"leave_one_clip_out": float(np.mean(clip_hits)), "leave_one_recording_out": float(np.mean(flat)),
                   "recording_out_clips": len(flat), "per_class_recording_out": {c: f"{sum(v)}/{len(v)}" for c, v in run_hits.items()},
                   "untestable_single_recording": sorted(set(untestable))}

meow_files, _, _ = catmeows("data/dataset")
M = [analyze(load(f), "ast")[1] for f in meow_files]
T = [analyze(load(f), "ast")[1] for f in files]
report["signals"] = {k: {"threshold": th,
                         "false_alarms_on_ordinary_meows": float(np.mean([s[k] >= th for s in M])),
                         "fires_on": {c: int(sum(T[i][k] >= th for i in np.where(y == c)[0])) for c in sorted(set(y))}}
                     for k, (*_, th) in SIGNALS.items()}

Path("benchmark/sound_types.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in ("beats", "ast")}, indent=1))
print({k: round(v["false_alarms_on_ordinary_meows"], 3) for k, v in report["signals"].items()})
