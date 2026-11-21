# Embed CatMeows under simulated "phone across the room" conditions: reverb, distance (gain), noise, inside a longer clip.
import sys, numpy as np
sys.path.insert(0, ".")
from meow_decoder import core, bench
files, y, cats, sessions = bench.catmeows("data/dataset")
rng = np.random.default_rng(0)
def room(w):
    rt = rng.uniform(0.2, 0.9); n = int(rt * 16000)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n); ir[0] = rng.uniform(1, 4); ir /= np.abs(ir).sum() ** 0.5 * 8
    x = np.convolve(w, ir)[: len(w) + n // 2]
    x = x / (np.abs(x).max() + 1e-9) * 10 ** (rng.uniform(-30, -6) / 20)
    x = bench.add_noise(x.astype(np.float32), rng.uniform(5, 30), rng)
    return bench.in_recording(x, rng, rng.uniform(2, 8)) if rng.random() < 0.5 else x
out = []
for i, f in enumerate(files):
    w = core.load(f)
    out.append([core.embedding(room(w), "beats") for _ in range(2)])
    if i % 50 == 0: print(i, flush=True)
np.save("data/wild/transfer_catmeows_far.npy", np.array(out))
