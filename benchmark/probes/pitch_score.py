import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from meow_decoder import bench
d = np.load("data/wild/pitch_features.npz"); names = list(d["names"])
C, y, cats, W = d["catmeows"], d["y"], d["cats"], d["wild"]; meta = json.loads(str(d["meta"]))
lab = np.array([m["label"] for m in meta])
print("NaN rows: catmeows", np.isnan(C).any(1).sum(), "wild", np.isnan(W).any(1).sum())
print("median per class (CatMeows):")
for k in "FBI": print(" ", k, dict(zip(names, np.nanmedian(C[y == k], 0).round(2))))
print("median per label (real):")
for k in ["F", "I", "B", "out", "wants_attention", "greet", "heat", "meow"]:
    if (lab == k).any(): print(" ", k, (lab == k).sum(), dict(zip(names, np.nanmedian(W[lab == k], 0).round(2))))
ok = ~np.isnan(C).any(1)
for f in range(len(names)):
    m = ok & np.isin(y, ["F", "I"]); mw = np.isin(lab, ["F", "I"]) & ~np.isnan(W).any(1)
    print(f"AUC food-vs-stressed by {names[f]:15s} CatMeows {roc_auc_score(y[m] == 'F', C[m, f]):.2f}  real {roc_auc_score(lab[mw] == 'F', W[mw, f]):.2f} (n={mw.sum()})")
# 3-class head on pitch only, unseen cats on CatMeows, then real
X = np.nan_to_num(C - np.nanmean(C, 0)) / np.nanstd(C, 0); Xw = np.nan_to_num(W - np.nanmean(C, 0)) / np.nanstd(C, 0)
P = bench.unseen_cat_proba(X, y, cats) if False else None
folds = bench.folds_for(cats); pred = np.empty(len(y), object)
for f in np.unique(folds):
    m = LogisticRegression(C=1, max_iter=2000).fit(X[folds != f], y[folds != f]); pred[folds == f] = m.predict(X[folds == f])
print("pitch-only 3-class, CatMeows unseen cats top1", (pred == y).mean().round(3))
m = LogisticRegression(C=1, max_iter=2000).fit(X, y); pw = m.predict(Xw); s = np.isin(lab, list("FBI"))
print("pitch-only on real meows: top1", (pw[s] == lab[s]).mean().round(2), {k: f"{(pw[s & (lab == k)] == k).sum()}/{(s & (lab == k)).sum()}" for k in "FBI"})
print("--- shape features only (rise, slope, peak position)")
sh = [names.index(n) for n in ["rise_st", "slope_st_per_s", "peak_pos"]]
mu, sd = np.nanmean(C[:, sh], 0), np.nanstd(C[:, sh], 0)
Xs, Xws = np.nan_to_num((C[:, sh] - mu) / sd), np.nan_to_num((W[:, sh] - mu) / sd)
for classes in (["F", "B", "I"], ["F", "I"]):
    m_ = np.isin(y, classes); pred = np.empty(m_.sum(), object)
    for f in np.unique(folds):
        t = folds[m_] == f
        pred[t] = LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(Xs[m_][~t], y[m_][~t]).predict(Xs[m_][t])
    mdl = LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(Xs[m_], y[m_]); s = np.isin(lab, classes); pw = mdl.predict(Xws)
    rec = {k: f"{(pw[s & (lab == k)] == k).sum()}/{(s & (lab == k)).sum()}" for k in classes}
    macro = np.mean([(pw[s & (lab == k)] == k).mean() for k in classes])
    print(f"{'/'.join(classes)}: CatMeows unseen cats {(pred == y[m_]).mean():.3f}; real top1 {(pw[s] == lab[s]).mean():.2f} macro {macro:.2f} {rec}")
