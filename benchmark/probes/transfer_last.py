import sys, zlib, numpy as np
sys.path.insert(0, ".")
from meow_decoder import core, bench
S = "data/wild/transfer_"
h = core.Head.load(core.weights_path("beats")); E = np.load(S + "wild_emb.npz"); far = np.load(S + "catmeows_far.npy")[:, 1]
X, y, cats = h.X, h.y, np.asarray(h.cats)
_, _, _, sessions = bench.catmeows("data/dataset")
lab, up = E["labels"], E["uploader"]; sit = np.isin(lab, list("FBI")); W, wl, wu = E["base"][sit], lab[sit], up[sit]
# 1. CatMeows + real meows from other uploaders, tested on each held-out uploader
pred = np.empty(len(W), object)
for u in np.unique(wu):
    t = wu == u
    pred[t] = core.Head.fit(np.vstack([X, W[~t]]), np.r_[y, wl[~t]]).classes[core.Head.fit(np.vstack([X, W[~t]]), np.r_[y, wl[~t]]).proba(W[t]).argmax(1)]
rec = [(pred[wl == c] == c).mean() for c in "FBI"]
print(f"1. +real meows, leave-one-uploader-out: top1 {(pred == wl).mean():.2f}, F/B/I recall {np.round(rec, 2)}, macro {np.mean(rec):.2f}")
# 2. Teaching under a microphone/room change: base head never saw the test cat; its meows come through a simulated room.
#    Teach 3 per situation from one session (room copies), test on the cat's other sessions (room copies).
folds = bench.folds_for(cats); before, after = [], []
for c in np.unique(cats):
    base = core.Head.fit(X[folds != folds[cats == c][0]], y[folds != folds[cats == c][0]])
    idx = np.where(cats == c)[0]
    for s in np.unique(sessions[idx]):
        teach_pool, test = idx[sessions[idx] == s], idx[sessions[idx] != s]
        rng = np.random.default_rng(zlib.crc32(f"{c}{s}".encode()))
        teach = [i for k in "FBI" for i in rng.permutation(teach_pool[y[teach_pool] == k])[:3]]
        if len(test) == 0 or len({y[i] for i in teach}) < 3: continue
        p = base.personalized([(far[i], y[i]) for i in teach])
        before += list(base.classes[base.proba(far[test]).argmax(1)] == y[test]); after += list(p.classes[p.proba(far[test]).argmax(1)] == y[test])
print(f"2. simulated room, different session: before teaching {np.mean(before):.2f}, after 3 per situation {np.mean(after):.2f} (n={len(before)})")
