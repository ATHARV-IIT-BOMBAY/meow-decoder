# Embed every real meow under several input treatments; save to npz for head experiments.
import json, sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "benchmark")
from meow_decoder import core, bench
from wild import labelled, local_name, WILD
rep = {o["title"]: o for o in json.load(open("benchmark/wild_report.json"))["clips_detail"]}
rows = [r for r in labelled() if rep[r["title"]]["has_meow"]]
V = {"base": lambda w: [w], "bl": lambda w: [bench.band_limit(w)],
     "seg_bl": lambda w: [bench.band_limit(w[a:b]) for a, b in bench.segments(w)[:1]],
     "seg3_bl": lambda w: [bench.band_limit(w[a:b]) for a, b in bench.segments(w)[:3]],
     "norm_bl": lambda w: [bench.normalize(bench.band_limit(w))]}
out = {k: [] for k in V}
for i, r in enumerate(rows):
    w = core.decode((WILD / "audio" / local_name(r)).read_bytes())
    for k, f in V.items():
        out[k].append(np.mean([core.embedding(x, "beats") for x in f(w)], 0))
    print(i, end=" ", flush=True)
np.savez("data/wild/transfer_wild_emb.npz",
         labels=np.array([r["label"] for r in rows]), uploader=np.array([r["artist"] for r in rows]), **{k: np.array(v) for k, v in out.items()})
