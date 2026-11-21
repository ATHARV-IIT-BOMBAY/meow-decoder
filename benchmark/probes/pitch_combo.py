"""Combine the BEATs head with the pitch-shape model; unseen cats on CatMeows and the real clips."""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, ".")
from meow_decoder import bench, core
d = np.load("data/wild/pitch_features.npz"); names = list(d["names"]); C, y, cats, W = d["catmeows"], d["y"], d["cats"], d["wild"]
meta = json.loads(str(d["meta"])); lab = np.array([m["label"] for m in meta])
rep = {o["title"]: o for o in json.load(open("benchmark/wild_report.json"))["clips_detail"]}
h = core.Head.load(core.weights_path("beats")); assert (h.y == y).all()
sh = [names.index(n) for n in ["rise_st", "slope_st_per_s", "peak_pos"]]
mu, sd = np.nanmean(C[:, sh], 0), np.nanstd(C[:, sh], 0)
Xs, Xws = np.nan_to_num((C[:, sh] - mu) / sd), np.nan_to_num((W[:, sh] - mu) / sd)
pitch = lambda Xtr, ytr: LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
folds = bench.folds_for(cats); Pp = np.zeros((len(y), 3))
for f in np.unique(folds):
    Pp[folds == f] = pitch(Xs[folds != f], y[folds != f]).predict_proba(Xs[folds == f])
Pb = bench.unseen_cat_proba(h.X, y, cats)
mp = pitch(Xs, y); assert list(mp.classes_) == list(h.classes)
s = np.array([m["title"] in rep and rep[m["title"]]["has_meow"] and m["label"] in "FBI" for m in meta])
Wb = np.array([[dict(rep[m["title"]]["results"])[k] for k in h.classes] if m["title"] in rep else [1/3] * 3 for m in meta])
Wp = mp.predict_proba(Xws)
def show(name, Pc, Pw):
    pc = h.classes[Pc.argmax(1)]; pw = h.classes[Pw.argmax(1)]
    rec = [(pw[s & (lab == k)] == k).mean() for k in "FBI"]
    print(f"{name:22s} CatMeows {np.mean(pc == y):.3f} macro {np.mean([(pc[y == k] == k).mean() for k in 'FBI']):.2f} | real top1 {np.mean(pw[s] == lab[s]):.2f} macro {np.mean(rec):.2f} F/B/I {[f'{(pw[s & (lab == k)] == k).sum()}/{(s & (lab == k)).sum()}' for k in 'FBI']}")
show("BEATs", Pb, Wb); show("pitch shape", Pp, Wp)
for a in (0.3, 0.5, 0.7):
    g = lambda B, P: np.exp(a * np.log(B + 1e-9) + (1 - a) * np.log(P + 1e-9))
    show(f"geo mix beats={a}", g(Pb, Pp), g(Wb, Wp))
