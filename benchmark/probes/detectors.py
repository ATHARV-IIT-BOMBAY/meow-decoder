"""Can a small trained detector beat AudioSet's own Hiss / Caterwaul / Purr scores? Grouped by uploader (and by cat for
CatMeows): a clip is always scored by a detector that never heard its uploader. Scored on the real-world clips, at the
same number of false alarms as the shipped threshold rule."""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, ".")
from meow_decoder import core
from transformers import AutoConfig

d = np.load("data/wild/detector_features.npz"); meta = json.loads(str(d["meta"]))
A, B = d["ast"], d["beats"]
lab = np.array([m["label"] for m in meta]); src = np.array([m["source"] for m in meta]); up = np.array([m["uploader"] for m in meta])
names = AutoConfig.from_pretrained(core.AST_ID, revision=core.AST_REVISION).id2label
col = {v: k for k, v in names.items()}
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
print("clips:", dict(zip(*np.unique(src, return_counts=True))), "| hiss", (lab == "hiss").sum(), "growl", (lab == "growl").sum(), "purr", (lab == "purr").sum())
TASKS = {"Hiss": ("hiss", {"human", "growl", "yowl"}), "Caterwaul": ("yowl", {"human", "hiss", "heat", "growl"}), "Purr": ("purr", {"human"})}
for key, (pos, ignore) in TASKS.items():
    use = ~np.isin(lab, list(ignore - {pos}))
    yb = lab == pos
    shipped = A[:, col[key]]
    for fname, X in [("AST 527 scores", logit(A)), ("BEATs embedding", B), ("both", np.hstack([logit(A), B]))]:
        score = np.full(len(yb), np.nan)
        for tr, te in GroupKFold(n_splits=10).split(X[use], yb[use], up[use]):
            tr, te = np.where(use)[0][tr], np.where(use)[0][te]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            m = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit((X[tr] - mu) / sd, yb[tr])
            score[te] = m.decision_function((X[te] - mu) / sd)
        w = use & (src == "wild")
        th_rule = core.SIGNALS[key][3]
        fa_rule = int((shipped[w & ~yb] >= th_rule).sum())
        # threshold matching the shipped rule's false alarms on the same real clips (ties: strict)
        negs = np.sort(score[w & ~yb])[::-1]
        th = negs[fa_rule] if fa_rule < len(negs) else -np.inf
        caught = int((score[w & yb] > th).sum())
        cm = use & (src == "catmeows")
        print(f"{key:9s} {fname:16s} AUC real {roc_auc_score(yb[w], score[w]):.2f} (shipped {roc_auc_score(yb[w], shipped[w]):.2f})"
              f" | at {fa_rule} false alarms: caught {caught}/{(w & yb).sum()} (shipped {(shipped[w & yb] >= th_rule).sum()})"
              f" | CatMeows false alarms {(score[cm] > th).mean():.1%} (shipped {(shipped[cm] >= th_rule).mean():.1%})")

print("--- threshold chosen inside each training fold (1% false alarms on its own out-of-fold negatives); BEATs embedding")
def fit(X, yb):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit((X - mu) / sd, yb)
    return lambda Q: m.decision_function((Q - mu) / sd)
for key, (pos, ignore) in TASKS.items():
    use = np.where(~np.isin(lab, list(ignore - {pos})))[0]; yb = lab == pos; shipped = A[:, col[key]]; th_rule = core.SIGNALS[key][3]
    hit = np.zeros(len(yb), bool)
    for tr, te in GroupKFold(n_splits=10).split(use, yb[use], up[use]):
        tr, te = use[tr], use[te]
        oof = np.full(len(tr), np.nan)
        for itr, ite in GroupKFold(n_splits=5).split(tr, yb[tr], up[tr]):
            oof[ite] = fit(B[tr[itr]], yb[tr[itr]])(B[tr[ite]])
        th = np.quantile(oof[~yb[tr]], 0.99)
        hit[te] = fit(B[tr], yb[tr])(B[te]) > th
    w = np.isin(np.arange(len(yb)), use) & (src == "wild"); cm = np.isin(np.arange(len(yb)), use) & (src == "catmeows")
    fs = np.isin(np.arange(len(yb)), use) & (src == "fsd50k")
    print(f"{key:9s} real: caught {hit[w & yb].sum()}/{(w & yb).sum()} false alarms {hit[w & ~yb].sum()}/{(w & ~yb).sum()}"
          f"  (shipped {(shipped[w & yb] >= th_rule).sum()}/{(w & yb).sum()}, {(shipped[w & ~yb] >= th_rule).sum()}/{(w & ~yb).sum()})"
          f" | FSD50K: caught {hit[fs & yb].sum()}/{(fs & yb).sum()} FA {hit[fs & ~yb].sum()}/{(fs & ~yb).sum()}"
          f" (shipped {(shipped[fs & yb] >= th_rule).sum()}, {(shipped[fs & ~yb] >= th_rule).sum()})"
          f" | CatMeows FA {hit[cm].mean():.1%} (shipped {(shipped[cm] >= th_rule).mean():.1%})")
