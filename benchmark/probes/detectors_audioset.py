"""AST mode: detectors on the 527 AudioSet scores (logits), trained only on CC0 / CC BY audio, tested on the real clips."""
import json, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "benchmark")
import detectors as D
from meow_decoder import core
from transformers import AutoConfig
clips = json.loads((D.DATA / "clips.json").read_text()); cache = np.load(D.DATA / "features.npz")
A = np.array([cache[f"ast|{c['id']}"] for c in clips]); X6 = None; X = np.log(np.clip(A, 1e-6, 1 - 1e-6) / (1 - np.clip(A, 1e-6, 1 - 1e-6)))
lab, src, up = (np.array([c[k] for c in clips]) for k in ("label", "source", "uploader"))
col = {v: int(k) for k, v in AutoConfig.from_pretrained(core.AST_ID, revision=core.AST_REVISION).id2label.items()}
train = (src != "wild") & np.array([D.commercial(c.get("license", "")) for c in clips]); test = src == "wild"
for key, (pos, ignore) in D.TASKS.items():
    yb, use = np.isin(lab, list(pos)), ~np.isin(lab, list(ignore)); t = train & use; w = test & use
    model, th = D.fit(X[t], yb[t]), D.threshold(X[t], yb[t], up[t])
    fired = D.score(model, X) > th; rule = A[:, col[key]] >= core.SIGNALS[key][3]
    nc = t & np.isin(lab, ["not_cat", "not_cat_hiss"])
    print(f"{key:9s} caught {fired[w & yb].sum()}/{(w & yb).sum()} FA {fired[w & ~yb].sum()}/{(w & ~yb).sum()} | AudioSet {rule[w & yb].sum()} / {rule[w & ~yb].sum()} | non-cat FA (in-sample) {fired[nc].sum()}/{nc.sum()} AudioSet {rule[nc].sum()}")
print("--- only the six scores the app keeps (Cat, Meow, Purr, Caterwaul, Hiss, Speech)")
six = [col[k] for k in core.CAT_LABELS + [core.SPEECH]]; X = X[:, six]
for key, (pos, ignore) in D.TASKS.items():
    yb, use = np.isin(lab, list(pos)), ~np.isin(lab, list(ignore)); t = train & use; w = test & use
    model, th = D.fit(X[t], yb[t]), D.threshold(X[t], yb[t], up[t])
    fired = D.score(model, X) > th; rule = A[:, col[key]] >= core.SIGNALS[key][3]
    print(f"{key:9s} caught {fired[w & yb].sum()}/{(w & yb).sum()} FA {fired[w & ~yb].sum()}/{(w & ~yb).sum()} | AudioSet {rule[w & yb].sum()} / {rule[w & ~yb].sum()}")
