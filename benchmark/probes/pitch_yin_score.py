import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from meow_decoder import bench, core
d = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/wild/pitch_yin.npz")
C, y, cats, W = d["catmeows"], d["y"], d["cats"], d["wild"]; meta = json.loads(str(d["meta"])); lab = np.array([m["label"] for m in meta])
rep = {o["title"]: o for o in json.load(open("benchmark/wild_report.json"))["clips_detail"]}
s = np.array([m["title"] in rep and rep[m["title"]]["has_meow"] and m["label"] in "FBI" for m in meta])
print("no pitch: CatMeows", np.isnan(C).any(1).sum(), "/", len(C), " real situation clips", np.isnan(W[s]).any(1).sum(), "/", s.sum())
for f, n in enumerate(core.PITCH_FEATURES):
    m = np.isin(y, ["F", "I"]) & ~np.isnan(C[:, f]); mw = s & np.isin(lab, ["F", "I"]) & ~np.isnan(W[:, f])
    print(f"AUC food>stressed {n:15s} CatMeows {roc_auc_score(y[m] == 'F', C[m, f]):.2f} real {roc_auc_score(lab[mw] == 'F', W[mw, f]):.2f}")
mu, sd = np.nanmean(C, 0), np.nanstd(C, 0); Z, Zw = np.nan_to_num((C - mu) / sd), np.nan_to_num((W - mu) / sd)
folds = bench.folds_for(cats)
for Creg in (0.1, 1.0):
    fit = lambda X, Y: LogisticRegression(C=Creg, max_iter=2000, class_weight="balanced").fit(X, Y)
    P = np.zeros((len(y), 3))
    for f in np.unique(folds):
        P[folds == f] = fit(Z[folds != f], y[folds != f]).predict_proba(Z[folds == f])
    pc = np.array(list("BFI"))[P.argmax(1)]; m = fit(Z, y); pw = m.classes_[m.predict_proba(Zw).argmax(1)]
    rec = [(pw[s & (lab == k)] == k).mean() for k in "FBI"]
    print(f"C={Creg}: CatMeows unseen cats {np.mean(pc == y):.3f} macro {np.mean([(pc[y == k] == k).mean() for k in 'FBI']):.2f} | real top1 {np.mean(pw[s] == lab[s]):.2f} macro {np.mean(rec):.2f} {[f'{(pw[s & (lab == k)] == k).sum()}/{(s & (lab == k)).sum()}' for k in 'FBI']}")
