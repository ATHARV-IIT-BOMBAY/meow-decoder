"""After teaching k meows per situation, is the taught BEATs head better than the untaught pitch guess? CatMeows
meows through a simulated room (as in transfer_farfield.py); taught meows from one session, tested on the cat's others."""
import sys, zlib
import numpy as np
sys.path.insert(0, ".")
from meow_decoder import bench, core
files, y, cats, sessions = bench.catmeows("data/dataset")
rng = np.random.default_rng(1)
def room(w):
    rt = rng.uniform(0.2, 0.9); n = int(rt * 16000)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n); ir[0] = rng.uniform(1, 4); ir /= np.abs(ir).sum() ** 0.5 * 8
    x = np.convolve(w, ir)[: len(w) + n // 2]
    x = x / (np.abs(x).max() + 1e-9) * 10 ** (rng.uniform(-30, -6) / 20)
    x = bench.add_noise(x.astype(np.float32), rng.uniform(5, 30), rng)
    return bench.in_recording(x, rng, rng.uniform(2, 8)) if rng.random() < 0.5 else x
cache = "data/wild/teach_vs_pitch.npz"
try:
    d = np.load(cache); E, P, PC = d["E"], d["P"], d["PC"]
except FileNotFoundError:
    E, P, PC = [], [], []
    for f in files:
        w = core.load(f); PC.append(core.pitch_features(w))
        r = room(w); E.append(core.embedding(r, "beats")); P.append(core.pitch_features(r))
    E, P, PC = np.array(E), np.array(P), np.array(PC); np.savez(cache, E=E, P=P, PC=PC)
X = core.Head.load(core.weights_path("beats")).X
folds = bench.folds_for(cats)
res = {k: [] for k in ["pitch (untaught)", "BEATs (untaught)", "taught 1", "taught 2", "taught 3", "taught 3 + pitch mix"]}
for c in np.unique(cats):
    tr = folds != folds[cats == c][0]
    base, ph = core.Head.fit(X[tr], y[tr]), core.fit_pitch_head(PC[tr], y[tr])
    idx = np.where(cats == c)[0]
    for s in np.unique(sessions[idx]):
        pool, test = idx[sessions[idx] == s], idx[sessions[idx] != s]
        if not len(test): continue
        r = np.random.default_rng(zlib.crc32(f"{c}{s}".encode()))
        order = {k: r.permutation(pool[y[pool] == k]) for k in "FBI"}
        hit = lambda Pr: list(np.array(["B", "F", "I"])[Pr.argmax(1)] == y[test])
        pp = ph.proba_features(P[test])
        res["pitch (untaught)"] += hit(pp); res["BEATs (untaught)"] += hit(base.proba(E[test]))
        for k in (1, 2, 3):
            taught = [(E[i], y[i]) for lab in "FBI" for i in order[lab][:k]]
            if not taught: continue
            pb = base.personalized(taught).proba(E[test])
            res[f"taught {k}"] += hit(pb)
            if k == 3: res["taught 3 + pitch mix"] += hit(np.sqrt(pb * pp))
for k, v in res.items(): print(f"{k:22s} {np.mean(v):.3f} (n={len(v)})")
