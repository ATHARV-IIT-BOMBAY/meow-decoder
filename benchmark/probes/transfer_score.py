import sys, numpy as np
sys.path.insert(0, ".")
from meow_decoder import core
E = np.load("data/wild/transfer_wild_emb.npz")
h = core.Head.load(core.weights_path("beats"))
lab = E["labels"]; sit = np.isin(lab, list("FBI"))
def report(name, P):
    pred = h.classes[P.argmax(1)]
    rec = {c: f"{(pred[sit & (lab == c)] == c).sum()}/{(sit & (lab == c)).sum()}" for c in "FBI"}
    macro = np.mean([(pred[sit & (lab == c)] == c).mean() for c in "FBI"])
    allp = {c: int((pred == c).sum()) for c in h.classes}
    print(f"{name:10s} top1 {(pred[sit] == lab[sit]).mean():.2f} macro {macro:.2f} {rec} all {allp}")
for k in ["base", "bl", "seg_bl", "seg3_bl", "norm_bl"]:
    report(k, h.proba(E[k]))
up = E["uploader"]
for k in ["base", "seg_bl"]:
    X = E[k]
    for mode in ["mean", "mean+std"]:
        P = np.zeros((len(X), 3))
        for u in np.unique(up):  # domain statistics from other uploaders' clips only
            o, t = up != u, up == u
            mu, sd = X[o].mean(0), X[o].std(0) + 1e-6
            Z = (X[t] - mu) / (h.scale if mode == "mean" else sd)
            P[t] = core.softmax(Z @ h.coef.T + h.intercept)
        report(f"{k}/{mode}", P)
