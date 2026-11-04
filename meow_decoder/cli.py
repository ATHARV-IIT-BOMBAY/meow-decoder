"""`meow` command line: predict, serve, train, benchmark, export-dataset."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from .core import ENCODERS

DATASET_CARD = """---
license: cc-by-4.0
task_categories: [audio-classification]
tags: [cats, bioacoustics, meow]
---
# Open cat meows

Meows recorded by Meow Decoder users who chose to share them, each labelled by the person who was with the cat.
`cat_id` is an anonymous hash that groups meows from the same cat: split by it, never at random, or
your test set will contain cats your model has already heard.

- 16 kHz mono wav, CC BY 4.0
- Labels: F = waiting for food, B = enjoying attention, I = stressed somewhere new, or the owner's own words
- Collected with https://github.com/ATHARV-IIT-BOMBAY/meow-decoder
"""


def main(argv=None):
    p = argparse.ArgumentParser(prog="meow", description="Guess what a cat's meow is about.")
    sub = p.add_subparsers(dest="cmd", required=True)
    enc = dict(choices=list(ENCODERS), default="beats", help="beats: most accurate, non-commercial license; ast: commercial-safe")

    s = sub.add_parser("predict", help="Rank the likely situations for audio files")
    s.add_argument("files", nargs="+")
    s.add_argument("--encoder", **enc)
    s.add_argument("--json", action="store_true", help="print JSON lines")

    s = sub.add_parser("serve", help="Run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--encoder", **enc)
    s.add_argument("--no-contrib", action="store_true", help="hide the 'share to the open dataset' option")

    s = sub.add_parser("train", help="Train the classifier on CatMeows (writes weights/<encoder>.npz)")
    s.add_argument("--data", default="data/dataset")
    s.add_argument("--encoder", **enc)
    s.add_argument("--out", default="weights")

    s = sub.add_parser("benchmark", help="Score on the fixed unseen-cat folds")
    s.add_argument("--data", default="data/dataset")
    s.add_argument("--encoder", **enc)

    s = sub.add_parser("export-dataset", help="Bundle meows you chose to share as a Hugging Face audio dataset")
    s.add_argument("out")

    a = p.parse_args(argv)
    try:
        return globals()[a.cmd.replace("-", "_")](a)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"meow: {e}")


def predict(a):
    from .core import Decoder
    d = Decoder(a.encoder)
    for f in a.files:
        r = d.predict(f)
        if a.json:
            print(json.dumps({"file": f, **r}, ensure_ascii=False))
            continue
        print(f"\n{f}" + ("" if r["is_cat"] else "  (doesn't sound much like a cat)"))
        for s in r["signals"]:
            print(f"  {s['emoji']} {s['title']}: {s['meaning']}")
        if not r["has_meow"]:
            print("  No meow in this clip, so no situation guess.")
            continue
        for x in r["results"]:
            print(f"  {x['p']:>4.0%}  {x['emoji']} {x['title']}: \"{x['line']}\"")


def serve(a):
    import uvicorn
    from .core import Decoder
    from .server import build_app, default_home
    import numpy as np
    print(f"Encoder: {ENCODERS[a.encoder]}\nLoading models…")
    d = Decoder(a.encoder)
    d.embed(np.zeros(16000, np.float32))  # load both audio models now, not on the first visitor's meow
    uvicorn.run(build_app(d, default_home(), contrib=not a.no_contrib), host=a.host, port=a.port)


def train(a):
    from .core import Head, catmeows, embed_files
    files, y, cats = catmeows(a.data)
    print(f"Embedding {len(files)} meows from {len(set(cats))} cats with {a.encoder}…")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Head.fit(embed_files(files, a.encoder), y, encoder=a.encoder).save(Path(a.out) / f"{a.encoder}.npz")
    print(f"Saved {Path(a.out) / f'{a.encoder}.npz'}")


def benchmark(a):
    from .core import benchmark as run, catmeows, embed_files
    files, y, cats = catmeows(a.data)
    r = run(embed_files(files, a.encoder), y, cats)
    print(json.dumps({"encoder": a.encoder, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}}, indent=1))


def export_dataset(a):
    from .server import default_home
    src, out = default_home() / "contrib", Path(a.out)
    rows = [json.loads(l) for l in (src / "metadata.jsonl").read_text().splitlines()] if (src / "metadata.jsonl").exists() else []
    if not rows:
        raise FileNotFoundError(f"No shared meows in {src}. Tick 'share' when you teach a meow in the app.")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            shutil.copy(src / r["file_name"], out / r["file_name"])
            w.writerow(r)
    (out / "README.md").write_text(DATASET_CARD)
    print(f"Wrote {len(rows)} meows to {out}. Upload with: huggingface-cli upload <you>/<dataset> {out} --repo-type dataset")


if __name__ == "__main__":
    main()
