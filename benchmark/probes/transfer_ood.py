import sys, numpy as np
sys.path.insert(0, ".")
from meow_decoder import core, bench
from sklearn.metrics import roc_auc_score
S = "data/wild/transfer_"
h = core.Head.load(core.weights_path("beats")); E = np.load(S + "wild_emb.npz")
X, cats = h.X, np.asarray(h.cats); far = np.load(S + "catmeows_far.npy")[:, 0]
def knn(train, Q, k=5):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    A, B = (train - mu) / sd, (Q - mu) / sd
    d = ((B[:, None] - A[None]) ** 2).sum(-1) ** 0.5
    return np.sort(d, 1)[:, :k].mean(1)
folds = bench.folds_for(cats)
held = np.zeros(len(X)); held_far = np.zeros(len(X))
for f in np.unique(folds):
    t = folds == f
    held[t] = knn(X[~t], X[t]); held_far[t] = knn(X[~t], far[t])
wild = knn(X, E["base"])
print("AUC unseen CatMeows cats vs real meows:", round(roc_auc_score(np.r_[np.zeros(len(X)), np.ones(len(wild))], np.r_[held, wild]), 3))
print("AUC unseen CatMeows vs CatMeows in simulated room:", round(roc_auc_score(np.r_[np.zeros(len(X)), np.ones(len(X))], np.r_[held, held_far]), 3))
for q in (0.95, 0.99, 1.0):
    th = np.quantile(held, q)
    print(f"threshold at {q:.0%} of unseen CatMeows cats: flags {(wild > th).mean():.0%} of real meows, {(held_far > th).mean():.0%} of simulated-room CatMeows")
print("held percentiles", np.percentile(held, [50, 95, 99, 100]).round(1), "wild", np.percentile(wild, [0, 5, 50]).round(1))
np.save(S + "held.npy", held)
