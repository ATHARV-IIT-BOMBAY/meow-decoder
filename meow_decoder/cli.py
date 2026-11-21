"""`meow` command line: predict, serve, train, benchmark, export-dataset, contribute."""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from .core import ENCODERS

DATASET_REPO = "AtharvIITB/open-cat-meows"
DATASET_CARD = """---
license: cc-by-4.0
task_categories: [audio-classification]
tags: [cats, bioacoustics, meow]
---
# Open cat meows

Meows recorded by Meow Decoder users who chose to share them, each labelled by the person who was with the cat.

**Split by `cat_id`, never at random**, or your test set will contain cats your model has already heard.
Better still, also keep each `session` on one side: meows from one visit share a room and a microphone.

| Column | Meaning |
|---|---|
| `label` | See below |
| `cat_id` | Anonymous hash: same cat, same id |
| `session` | Anonymous hash of one app visit |
| `device` | `phone` or `computer` |
| `hour` | Local hour of the day, if known |
| `where` | `home`, `outdoors`, `vet`, `car`, `other`, or empty |
| `contribution_id` | Quote it to have a meow removed |

Labels: `F` waiting for food, `B` enjoying attention, `I` stressed somewhere new, `greet` saying hello,
`out` wants to go out, `play` wants to play, `annoyed` annoyed, or `custom` (the owner typed their own words;
the text is only included after review).

- 16 kHz mono wav, CC BY 4.0. Clips with talking in them, or that don't sound like a cat, are not accepted.
- To remove a meow, open an issue at https://github.com/ATHARV-IIT-BOMBAY/meow-decoder with its `contribution_id`.
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
    s.add_argument("--data", default="data/dataset", help="CatMeows folder; without it, scores the stored embeddings in the weights")
    s.add_argument("--encoder", **enc)
    s.add_argument("--from-weights", action="store_true", help="use the embeddings stored in the weights (no audio or models needed)")
    s.add_argument("--teaching", action="store_true", help="also score teaching a cat, across recording sessions (needs --data)")
    s.add_argument("--robust", action="store_true", help="also score real-world recording conditions (needs --data, a few minutes)")
    s.add_argument("--predictions", metavar="CSV", help="score your own model: columns file_name, F, B, I (probabilities)")

    s = sub.add_parser("export-dataset", help="Bundle meows you chose to share as a Hugging Face audio dataset")
    s.add_argument("out")
    s.add_argument(
        "--include-custom-labels", action="store_true", help="keep the owner's own words for 'custom' labels (check them for names first)"
    )

    sub.add_parser("contribute", help=f"Send the meows you shared to the open dataset ({DATASET_REPO}) as a pull request")

    a = p.parse_args(argv)
    try:
        return globals()[a.cmd.replace("-", "_")](a)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"meow: {e}")


def predict(a):
    from .core import Decoder

    d = Decoder(a.encoder)
    failed = 0
    for f in a.files:
        try:
            r = d.predict(f)
        except (OSError, ValueError) as e:  # keep going: one bad file shouldn't stop the rest
            failed += 1
            print(f"\n{f}\n  error: {e}", file=sys.stderr)
            continue
        if a.json:
            print(json.dumps({"file": f, **r}, ensure_ascii=False))
            continue
        print(f"\n{f}" + ("" if r["is_cat"] else "  (doesn't sound much like a cat)"))
        for s in r["signals"]:
            print(f"  {s['emoji']} {s['title']}: {s['meaning']}")
        if not r["has_meow"]:
            print("  No meow in this clip, so no situation guess." if r["is_cat"] else "  No situation guess.")
            continue
        top, second = r["results"][:2]
        if top["p"] < 0.5:
            print(f"  Not sure: it's between {top['title'].lower()} and {second['title'].lower()}.")
        for x in r["results"]:
            print(f'  {x["p"]:>4.0%}  {x["emoji"]} {x["title"]}: "{x["line"]}"')
    if not a.json:
        print(
            "\nSituation percentages are from cats recorded in a lab. On real home recordings the first guess was right"
            " about 1 time in 7 (README: Tested on real recordings). Purring, hissing and yowling are more reliable.",
            file=sys.stderr,
        )
    if failed:
        sys.exit(f"meow: {failed} of {len(a.files)} files couldn't be read")


def serve(a):
    import numpy as np
    import uvicorn

    from .core import Decoder
    from .server import build_app, default_home

    try:
        local = ipaddress.ip_address(a.host).is_loopback
    except ValueError:
        local = a.host == "localhost"
    print(f"Encoder: {ENCODERS[a.encoder]}")
    if not local:
        print(
            "Warning: anyone who can reach this address can use the app. There are no logins; each browser only sees "
            "its own cats. For the internet, put it behind https (e.g. cloudflared) and see 'Deploy it' in the README."
        )
    print("Loading models…")
    d = Decoder(a.encoder)
    d.embed(np.zeros(16000, np.float32))  # load both audio models now, not on the first visitor's meow
    uvicorn.run(build_app(d, default_home(), contrib=not a.no_contrib, require_owner=not local), host=a.host, port=a.port)


def train(a):
    from .bench import catmeows, embed_dataset
    from .core import Head

    files, y, cats, _ = catmeows(a.data)
    print(f"Embedding {len(files)} meows from {len(set(cats))} cats with {a.encoder}…")
    X, yy, cc, train_only = embed_dataset(files, y, cats, a.encoder)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    h = Head.fit(X, yy, encoder=a.encoder)
    h.cats, h.train_only = cc, train_only
    h.save(Path(a.out) / f"{a.encoder}.npz")
    print(f"Saved {Path(a.out) / f'{a.encoder}.npz'}")


def benchmark(a):
    from .bench import CONDITIONS, catmeows, embed_dataset, robustness, teaching_curve
    from .bench import benchmark as run
    from .core import Head, weights_path

    rnd = lambda v: round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] if isinstance(v, list) else v
    if a.predictions:
        print(json.dumps({"predictions": a.predictions, **{k: rnd(v) for k, v in score_predictions(a.predictions).items()}}, indent=1))
        return
    if a.from_weights or not Path(a.data).is_dir():
        h = Head.load(weights_path(a.encoder))
        if h.cats is None:
            raise ValueError("These weights don't store cat ids. Use --data with the CatMeows folder.")
        n = int((~h.train_only).sum()) if h.train_only is not None else len(h.y)
        print(f"Scoring the {n} embeddings stored in the {a.encoder} weights")
        r = run(h.X, h.y, h.cats, h.train_only)
        print(json.dumps({"encoder": a.encoder, **{k: rnd(v) for k, v in r.items()}}, indent=1))
        return
    files, y, cats, sessions = catmeows(a.data)
    X, yy, cc, train_only = embed_dataset(files, y, cats, a.encoder)
    r = run(X, yy, cc, train_only)
    print(json.dumps({"encoder": a.encoder, **{k: rnd(v) for k, v in r.items()}}, indent=1))
    if a.teaching:
        clean = ~train_only
        print("\nTeaching one cat, tested on its other recording sessions:")
        for row in teaching_curve(X[clean], y, cats, sessions):
            right, conf = f"{row['accuracy']:.1%} right", f"{row['confidence']:.1%} average confidence"
            print(f"  {row['k']} per situation: {right}, {conf} ({row['tested']} tests)")
    if a.robust:
        print("\nReal-world recording conditions (models trained as `meow train` does):")
        print(f"  {'Clean dataset files':40s} {r['top1']:.1%}")
        for name, row in robustness(files, y, cats, a.encoder).items():
            print(f"  {CONDITIONS[name][0]:40s} {row['top1']:.1%}  (calibration error {row['ece']:.1%})")


def score_predictions(path):
    """Score another model's out-of-fold probabilities for the 440 CatMeows clips, on the same metrics as ours."""
    import numpy as np

    from .bench import metrics

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    names = [r["file_name"].rsplit("/", 1)[-1] for r in rows]
    if len(set(names)) != 440 or not all(n.endswith(".wav") and n[0] in "FBI" for n in names):
        raise ValueError("Expected one row for each of the 440 CatMeows files (file_name like B_ANI01_MC_FN_SIM01_101.wav).")
    P = np.array([[float(r[k]) for k in ("B", "F", "I")] for r in rows])  # classes in sorted order
    y = np.array([n[0] for n in names])
    cats = np.array([n.split("_")[1] for n in names])
    return metrics(P / P.sum(1, keepdims=True), y, cats)


def export_dataset(a, out=None):
    from .server import default_home

    src, out = default_home() / "contrib", Path(out or a.out)
    rows = [json.loads(line) for line in (src / "metadata.jsonl").read_text().splitlines()] if (src / "metadata.jsonl").exists() else []
    if not rows:
        raise FileNotFoundError(f"No shared meows in {src}. Tick 'share' when you teach a meow in the app.")
    keep_text = getattr(a, "include_custom_labels", False)
    fields = ["file_name", "contribution_id", "label", "cat_id", "session", "device", "hour", "where", "seconds", "submitted", "license"]
    fields += ["label_text"] if keep_text else []
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            shutil.copy(src / r["file_name"], out / r["file_name"])
            w.writerow(r)
    (out / "README.md").write_text(DATASET_CARD)
    custom = sum(r.get("label") == "custom" for r in rows)
    note = (
        f" {custom} have your own words as labels; they're exported as 'custom' unless you pass --include-custom-labels." if custom else ""
    )
    print(f"Wrote {len(rows)} meows to {out}.{note}")
    return len(rows)


def contribute(a):
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        n = export_dataset(a, Path(tmp) / "bundle")
        folder = f"contributions/{uuid.uuid4().hex[:12]}"
        (Path(tmp) / "bundle" / "README.md").unlink()  # the dataset has one card at its root
        try:
            info = HfApi().upload_folder(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                folder_path=str(Path(tmp) / "bundle"),
                path_in_repo=folder,
                create_pr=True,
                commit_message=f"Add {n} meows",
            )
        except Exception as e:
            raise ValueError(
                f"Upload failed ({e}). Log in first with `huggingface-cli login`, or upload "
                f"`meow export-dataset` output yourself to https://huggingface.co/datasets/{DATASET_REPO}."
            ) from e
    print(f"Opened a pull request with {n} meows: {info.pr_url or info.commit_url}\nThank you! Maintainers review it before merging.")


if __name__ == "__main__":
    main()
