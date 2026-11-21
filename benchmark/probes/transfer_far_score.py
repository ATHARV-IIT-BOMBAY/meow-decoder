import sys, numpy as np
sys.path.insert(0, ".")
from meow_decoder import core, bench
S = "data/wild/transfer_"
h = core.Head.load(core.weights_path("beats")); E = np.load(S + "wild_emb.npz"); far = np.load(S + "catmeows_far.npy")
files, y, cats, _ = bench.catmeows("data/dataset")
assert len(h.X) == 440 and (h.y == y).all() and (np.asarray(h.cats) == cats).all()
lab = E["labels"]; sit = np.isin(lab, list("FBI"))
def wild(head, X):
    pred = head.classes[head.proba(X).argmax(1)]
    rec = [(pred[sit & (lab == c)] == c).mean() for c in "FBI"]
    return f"wild top1 {(pred[sit] == lab[sit]).mean():.2f} macro {np.mean(rec):.2f} F/B/I {[round(r,2) for r in rec]} F-calls {(pred == 'F').sum()}/121"
for k in (1, 2):
    Xa = np.vstack([h.X] + [far[:, i] for i in range(k)]); ya = np.tile(y, k + 1); ca = np.tile(cats, k + 1)
    to = np.r_[np.zeros(440, bool), np.ones(440 * k, bool)]
    m = bench.metrics(bench.unseen_cat_proba(Xa, ya, ca, train_only=to)[~to] if False else None, y, cats) if False else None
    P = bench.unseen_cat_proba(Xa, ya, ca, train_only=to)
    P = P[: 440] if len(P) == len(ya) else P
    mt = bench.metrics(P, y, cats)
    print(f"clean+{k} far copies: CatMeows unseen-cat top1 {mt['top1']:.3f}", wild(core.Head.fit(Xa, ya), E["base"]))
    # far copies only (no clean)
Xf = far.reshape(-1, far.shape[-1]); yf = np.repeat(y, 2)
print("far-only train:", wild(core.Head.fit(Xf, yf), E["base"]))
print("baseline:", f"CatMeows {bench.metrics(bench.unseen_cat_proba(h.X, y, cats), y, cats)['top1']:.3f}", wild(h, E["base"]))
