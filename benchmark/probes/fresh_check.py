"""Fresh check: FSD50K meows (not in wild_labels.csv) whose descriptions state the situation, labelled before scoring."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, ".")
from huggingface_hub import hf_hub_download
from meow_decoder import bench, core
FRESH = {"55152": ("F", "Waiting for food, they annoy me audibly"), "220018": ("F", "what took you so long to come outside and feed me"),
         "220019": ("F", "what took you so long to come outside and feed me"), "76589": ("F", "Cat wants food"), "76590": ("F", "Cat wants food"),
         "149191": ("I", "kittens meowing ... taken in a vet clinic"), "232343": ("I", "little cat, that is alone and scarry")}  # fmt: skip
split = {r: "dev" for r in FRESH}; split.update({k: "eval" for k in []})
d = np.load("data/wild/pitch_yin.npz"); C, y = d["catmeows"], d["y"]
mu, sd = np.nanmean(C, 0), np.nanstd(C, 0)
m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(np.nan_to_num((C - mu) / sd), y)
dec = core.Decoder("beats")
for fid, (lab, why) in FRESH.items():
    for sp in ("dev", "eval"):
        try:
            p = hf_hub_download("Fhrozen/FSD50k", f"clips/{sp}/{fid}.wav", repo_type="dataset"); break
        except Exception:
            continue
    w = core.decode(open(p, "rb").read())
    pf = core.pitch_features(w); pp = m.predict_proba(np.nan_to_num((pf[None] - mu) / sd))[0]
    r = dec.predict(p)
    print(f"{fid:7s} {lab}  pitch -> {m.classes_[pp.argmax()]} ({pp.max():.2f})  BEATs -> {r['results'][0]['key']} ({r['results'][0]['p']:.2f})  has_meow {r['has_meow']}  rise {pf[0]:.1f}  [{why}]")
