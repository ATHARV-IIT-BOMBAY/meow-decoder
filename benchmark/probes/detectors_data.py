"""Features for the purr/hiss/yowl detector experiments: AST AudioSet scores (all 527 classes, max over 10 s windows)
and the BEATs embedding, for the real-world clips, FSD50K's cat clips and CatMeows. -> data/wild/detector_features.npz

FSD50K (CC licences, labels checked by people) is fetched from the Hugging Face mirror Fhrozen/FSD50k.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "benchmark")
from wild import WILD, labelled, local_name  # noqa: E402

from meow_decoder import bench, core  # noqa: E402

FSD = Path("data/fsd50k")


def fsd_cat_clips():
    from huggingface_hub import hf_hub_download

    info = {**json.load(open(FSD / "dev_clips_info_FSD50K.json")), **json.load(open(FSD / "eval_clips_info_FSD50K.json"))}
    rows = [(r, "dev") for r in csv.DictReader(open(FSD / "dev.csv"))] + [(r, "eval") for r in csv.DictReader(open(FSD / "eval.csv"))]
    out = []
    for r, split in rows:
        L = set(r["labels"].split(","))
        if "Cat" not in L:
            continue
        label = "hiss" if "Hiss" in L else "growl" if "Growling" in L else "purr" if "Purr" in L else "meow" if "Meow" in L else "cat"
        path = hf_hub_download("Fhrozen/FSD50k", f"clips/{split}/{r['fname']}.wav", repo_type="dataset")
        meta = info[r["fname"]]
        out.append({"id": f"fsd:{r['fname']}", "freesound_id": r["fname"], "label": label, "uploader": meta["uploader"], "path": path,
                    "source": "fsd50k", "labels": r["labels"]})  # fmt: skip
    return out


def ast_all(wave):
    import torch

    fe, ast, dev, _ = core._ast()
    windows = [wave[i : i + 10 * core.SR] for i in range(0, max(len(wave) - core.SR // 2, 1), 10 * core.SR)]
    with torch.no_grad():
        logits = torch.cat([ast(**fe(w, sampling_rate=core.SR, return_tensors="pt").to(dev)).logits for w in windows])
    return torch.sigmoid(logits).max(0).values.cpu().numpy()


def main():
    clips = [{"id": "wild:" + r["title"], "freesound_id": r["title"].split()[1].rstrip(":") if r["title"].startswith("Freesound") else "",
              "label": r["label"], "uploader": r["artist"], "path": str(WILD / "audio" / local_name(r)), "source": "wild", "labels": ""}
             for r in labelled()]  # fmt: skip
    test_ids = {c["freesound_id"] for c in clips if c["freesound_id"]}
    clips += [c for c in fsd_cat_clips() if c["freesound_id"] not in test_ids]
    files, y, cats, _ = bench.catmeows("data/dataset")
    clips += [{"id": "catmeows:" + f.name, "freesound_id": "", "label": "catmeows_" + l, "uploader": "catmeows:" + c, "path": str(f),
               "source": "catmeows", "labels": ""} for f, l, c in zip(files, y, cats)]  # fmt: skip
    A, B = [], []
    for i, c in enumerate(clips):
        p = Path(c["path"])
        w = core.load(p) if p.suffix == ".wav" and c["source"] == "catmeows" else core.decode(p.read_bytes())
        A.append(ast_all(w))
        B.append(core.embedding(w, "beats"))
        if i % 50 == 0:
            print(i, len(clips), flush=True)
    np.savez("data/wild/detector_features.npz", ast=np.array(A), beats=np.array(B), meta=json.dumps(clips))


if __name__ == "__main__":
    main()
