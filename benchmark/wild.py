"""Real-world test set: openly licensed cat recordings from the internet, labelled from what the uploader said.

    python benchmark/wild.py search     # list candidate files on Wikimedia Commons -> data/wild/candidates.json
    python benchmark/wild.py download   # fetch the clips labelled in benchmark/wild_labels.csv -> data/wild/audio/
    python benchmark/wild.py evaluate   # score the decoder on them -> benchmark/wild_report.json

Only openly licensed sources are used (Wikimedia Commons: CC0, public domain, CC BY, CC BY-SA). Audio is never
committed: wild_labels.csv records each file's source, author, licence and the words in its description that
justify the label, and `download` streams the first 30 s of audio from the source.

These labels are weaker than CatMeows': they come from what an uploader wrote, not from a controlled setup, so
this is a test set only, never training data.
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

UA = "MeowDecoderResearch/0.4 (https://github.com/ATHARV-IIT-BOMBAY/meow-decoder)"
API = "https://commons.wikimedia.org/w/api.php"
OPEN = re.compile(r"^(CC0|Public domain|PD|CC BY(-SA)? [0-9.]+|CC BY(-SA)?|GFDL|Attribution)", re.I)
CATEGORIES = ["Category:Audio files of cats", "Category:Audio files of cats meowing"]
QUERIES = [
    "meow",
    "meowing",
    "purr",
    "purring",
    "cat hiss",
    "cat yowl",
    "Felis catus",
    "Katze",
    "miau",
    "miaou",
    "maullido",
    "miagolio",
    "kitten",
    "cat crying",
    "cat fight",
]
LABELS = Path(__file__).with_name("wild_labels.csv")
WILD = Path("data/wild")


def api(**params):
    params["format"] = "json"
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=30))
        except OSError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Commons API failed for {params}")


def search(_):
    titles = set()
    for cat in CATEGORIES:
        cont = {}
        while True:
            d = api(action="query", list="categorymembers", cmtitle=cat, cmtype="file", cmlimit=500, **cont)
            titles |= {m["title"] for m in d["query"]["categorymembers"]}
            if "continue" not in d:
                break
            cont = {"cmcontinue": d["continue"]["cmcontinue"]}
    for kind in ("audio", "video"):
        for q in QUERIES:
            d = api(action="query", list="search", srsearch=f"{q} filetype:{kind}", srnamespace=6, srlimit=200)
            titles |= {x["title"] for x in d["query"]["search"]}
        print(f"{kind}: {len(titles)} candidates so far", flush=True)
    strip = lambda h: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h or "")).strip()
    rows, titles = [], sorted(titles)
    for i in range(0, len(titles), 40):
        d = api(action="query", titles="|".join(titles[i : i + 40]), prop="imageinfo", iiprop="url|size|mime|extmetadata")
        for p in d["query"]["pages"].values():
            ii = (p.get("imageinfo") or [{}])[0]
            m = ii.get("extmetadata", {})
            val = lambda k, m=m: strip(m.get(k, {}).get("value"))
            rows.append({"title": p["title"], "url": ii.get("url"), "page": ii.get("descriptionurl"), "mime": ii.get("mime"),
                         "bytes": ii.get("size"), "license": val("LicenseShortName"), "artist": val("Artist"),
                         "description": val("ImageDescription")[:1500], "categories": val("Categories")[:600]})  # fmt: skip
    rows = [r for r in rows if OPEN.match(r["license"] or "")]
    WILD.mkdir(parents=True, exist_ok=True)
    (WILD / "candidates.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"{len(rows)} openly licensed candidates -> {WILD / 'candidates.json'}")


def labelled():
    with open(LABELS, newline="") as f:
        return list(csv.DictReader(f))


def local_name(row):
    """Readable and unique, even on case-insensitive file systems ("Cat Purring" vs "Cat purring")."""
    stem = re.sub(r"[^\w.-]+", "_", row["title"].removeprefix("File:")).rsplit(".", 1)[0]
    return f"{stem}_{zlib.crc32(row['title'].encode()):08x}.wav"


def download(_):
    """Stream only the first 30 s of audio from each labelled file (what the decoder uses) into a 16 kHz wav."""
    import subprocess

    out = WILD / "audio"
    out.mkdir(parents=True, exist_ok=True)
    for row in labelled():
        dest = out / local_name(row)
        if dest.exists():
            continue
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-user_agent",
            UA,
            "-i",
            row["url"],
            "-t",
            "30",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dest),
        ]
        for attempt in range(4):
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 or "429" not in r.stderr:
                break
            time.sleep(30 * (attempt + 1))  # rate-limited: back off
        print("downloaded" if r.returncode == 0 else f"FAILED ({r.stderr.strip()[-120:]})", dest.name, flush=True)
        time.sleep(4)  # Wikimedia rate-limits fast downloaders


def evaluate(a):
    import numpy as np

    from meow_decoder.core import Decoder

    d = Decoder(a.encoder)
    rows, out = labelled(), []
    for row in rows:
        r = d.predict(WILD / "audio" / local_name(row))
        out.append({"title": row["title"], "label": row["label"], "is_cat": r["is_cat"], "has_meow": r["has_meow"], "sounds": r["sounds"],
                    "signals": [s["key"] for s in r["signals"]], "results": [(x["key"], x["p"]) for x in r["results"]]})  # fmt: skip
    situation = [o for o in out if o["label"] in ("F", "B", "I")]
    scored = [o for o in situation if o["has_meow"]]
    hit = [o["results"][0][0] == o["label"] for o in scored]
    top2 = [o["label"] in (o["results"][0][0], o["results"][1][0]) for o in scored]
    conf = [o["results"][0][1] for o in scored]
    cats = [o for o in out if o["label"] != "human"]

    def detector(key, label, among=None):  # caught / labelled, and false alarms on clips with other labels
        pos = [o for o in out if o["label"] == label]
        neg = [o for o in out if o["label"] not in (label, *(among or ()))]
        return {"caught": f"{sum(key in o['signals'] for o in pos)}/{len(pos)}",
                "false_alarms": f"{sum(key in o['signals'] for o in neg)}/{len(neg)}"}  # fmt: skip

    report = {
        "encoder": a.encoder,
        "clips": len(out),
        "situations": {
            "labelled": len(situation),
            "scored": len(scored),
            "top1": float(np.mean(hit)) if hit else None,
            "top2": float(np.mean(top2)) if top2 else None,
            "confidence": float(np.mean(conf)) if conf else None,
            "majority": max((np.mean([o["label"] == c for o in scored]) for c in "FBI"), default=None),
            "by_label": {
                c: f"{sum(h for o, h in zip(scored, hit) if o['label'] == c)}/{sum(o['label'] == c for o in scored)}" for c in "FBI"
            },
        },  # fmt: skip
        "cat_detected_on_cats": f"{sum(o['is_cat'] for o in cats)}/{len(cats)}",
        "humans_called_cat": f"{sum(o['is_cat'] for o in out if o['label'] == 'human')}/{sum(o['label'] == 'human' for o in out)}",
        "predicted_stressed_share": float(np.mean([o["results"][0][0] == "I" for o in out if o["has_meow"]])),
        "purr": detector("Purr", "purr"),
        "hiss": detector("Hiss", "hiss"),
        "yowl": detector("Caterwaul", "yowl", among=("hiss", "heat")),  # growls and heat calls are yowl-like
        "clips_detail": out,
    }
    Path(__file__).with_name("wild_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "clips_detail"}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("search")
    sub.add_parser("download")
    e = sub.add_parser("evaluate")
    e.add_argument("--encoder", default="beats")
    a = ap.parse_args()
    {"search": search, "download": download, "evaluate": evaluate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
