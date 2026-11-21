"""Real-world test set: openly licensed cat recordings from the internet, labelled from what the uploader said.

    python benchmark/wild.py search                      # Wikimedia Commons -> data/wild/candidates.json
    python benchmark/wild.py search --source freesound   # Freesound -> data/wild/freesound_candidates.json
    python benchmark/wild.py download   # fetch the clips labelled in benchmark/wild_labels.csv -> data/wild/audio/
    python benchmark/wild.py evaluate   # score the decoder on them -> benchmark/wild_report.json

Only openly licensed sources are used: Wikimedia Commons (CC0, public domain, CC BY, CC BY-SA) and Freesound
(CC0, CC BY, CC BY-NC; this is non-commercial research and no audio is redistributed). Freesound needs a free API
key in FREESOUND_API_KEY (environment or a .env file). Audio is never committed: wild_labels.csv records each
clip's source, uploader, licence and the words in its description that justify the label, and `download` streams
the first 30 s of audio (Freesound's public MP3 preview) from the source.

Several clips often come from one uploader, so usually one cat; `evaluate` reports how many uploaders each result
rests on.

These labels are weaker than CatMeows': they come from what an uploader wrote, not from a controlled setup, so
this is a test set only, never training data.
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
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
FREESOUND = "https://freesound.org/apiv2/search/text/"
FREESOUND_QUERIES = [  # situations first: few uploaders say why a cat meowed, so ask for it directly
    "cat hungry", "cat food meow", "cat begging", "cat feeding time", "cat breakfast", "cat dinner", "cat demanding food",
    "cat petting", "cat purring petting", "cat brushing", "cat cuddle", "cat attention", "cat belly rub",
    "cat vet", "cat carrier", "cat car", "cat travel", "cat new home", "cat moving", "cat shelter", "cat scared",
    "cat door", "cat wants out", "cat wants in", "cat outside", "cat locked", "cat window",
    "cat hiss", "cat growl", "cat fight", "cat yowl", "tomcat", "cat in heat", "cat mating",
    "cat purr", "cat meow", "kitten meow", "kitten crying", "cat chirp", "cat trill", "cat greeting", "cat play",
    "cat crying", "cat calling", "cat lonely", "cat alone", "cat morning", "cat annoyed",
]  # fmt: skip
LICENSES = [("publicdomain/zero", "CC0"), ("licenses/by-nc", "CC BY-NC"), ("licenses/by/", "CC BY"), ("sampling+", "Sampling+")]
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


def freesound_key():
    import os

    key = os.environ.get("FREESOUND_API_KEY")
    env = Path(".env")
    if not key and env.exists():
        key = dict(line.strip().split("=", 1) for line in env.read_text().splitlines() if "=" in line).get("FREESOUND_API_KEY")
    if not key:
        sys.exit("Set FREESOUND_API_KEY (get a free key at https://freesound.org/apiv2/apply/).")
    return key.strip().strip("'\"")


def search_freesound():
    key, found = freesound_key(), {}
    fields = "id,name,description,tags,license,username,duration,url,previews"
    for q in FREESOUND_QUERIES:
        for page in range(1, 4):
            url = FREESOUND + "?" + urllib.parse.urlencode({"query": q, "page": page, "page_size": 150, "fields": fields})
            req = urllib.request.Request(url, headers={"Authorization": f"Token {key}", "User-Agent": UA})
            try:
                d = json.load(urllib.request.urlopen(req, timeout=30))
            except urllib.error.HTTPError as e:
                if e.code == 404:  # past the last page
                    break
                raise
            for r in d["results"]:
                lic = next((short for part, short in LICENSES if part in r["license"].lower()), r["license"])
                found[r["id"]] = {"title": f"Freesound {r['id']}: {r['name']}", "url": r["previews"]["preview-hq-mp3"], "page": r["url"],
                                  "license": lic, "artist": r["username"], "description": re.sub(r"\s+", " ", r["description"])[:1500],
                                  "categories": " ".join(r["tags"])[:600], "seconds": r["duration"], "source": "freesound"}  # fmt: skip
            time.sleep(1.1)  # Freesound allows 60 requests a minute
            if not d.get("next"):
                break
        print(f"{q!r}: {len(found)} clips so far", flush=True)
    WILD.mkdir(parents=True, exist_ok=True)
    (WILD / "freesound_candidates.json").write_text(json.dumps(list(found.values()), indent=1, ensure_ascii=False))
    print(f"{len(found)} Freesound candidates -> {WILD / 'freesound_candidates.json'}")


def search(a):
    if a.source == "freesound":
        return search_freesound()
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
        time.sleep(4 if "wikimedia" in row["url"] else 0.5)  # Wikimedia rate-limits fast downloaders


def evaluate(a):
    import numpy as np

    from meow_decoder.core import Decoder

    d = Decoder(a.encoder)
    rows, out = labelled(), []
    for row in rows:
        r = d.predict(WILD / "audio" / local_name(row))
        out.append({"title": row["title"], "label": row["label"], "uploader": row["artist"],
                    "source": "freesound" if "freesound.org" in row["page"] else "commons",
                    "is_cat": r["is_cat"], "has_meow": r["has_meow"], "sounds": r["sounds"],
                    "signals": [s["key"] for s in r["signals"]], "results": [(x["key"], x["p"]) for x in r["results"]]})  # fmt: skip
    situation = [o for o in out if o["label"] in ("F", "B", "I")]
    scored = [o for o in situation if o["has_meow"]]
    hit = [o["results"][0][0] == o["label"] for o in scored]
    top2 = [o["label"] in (o["results"][0][0], o["results"][1][0]) for o in scored]
    conf = [o["results"][0][1] for o in scored]
    cats = [o for o in out if o["label"] != "human"]

    def detector(key, labels, among=()):  # caught / labelled, and false alarms on clips with other labels
        pos = [o for o in out if o["label"] in labels]
        neg = [o for o in out if o["label"] not in (*labels, *among)]
        return {"caught": f"{sum(key in o['signals'] for o in pos)}/{len(pos)}",
                "false_alarms": f"{sum(key in o['signals'] for o in neg)}/{len(neg)}"}  # fmt: skip

    def situations(scored):
        hit = [o["results"][0][0] == o["label"] for o in scored]
        return {"scored": len(scored), "uploaders": len({o["uploader"] for o in scored}), "top1": float(np.mean(hit)) if hit else None,
                "top2": float(np.mean([o["label"] in dict(o["results"][:2]) for o in scored])) if scored else None}  # fmt: skip

    report = {
        "encoder": a.encoder,
        "clips": len(out),
        "situations": {
            "labelled": len(situation),
            "scored": len(scored),
            "top1": float(np.mean(hit)) if hit else None,
            "top2": float(np.mean(top2)) if top2 else None,
            "confidence": float(np.mean(conf)) if conf else None,
            "uploaders": len({o["uploader"] for o in scored}),
            "by_source": {src: situations([o for o in scored if o["source"] == src]) for src in ("commons", "freesound")},
            "majority": max((np.mean([o["label"] == c for o in scored]) for c in "FBI"), default=None),
            "by_label": {
                c: f"{sum(h for o, h in zip(scored, hit) if o['label'] == c)}/{sum(o['label'] == c for o in scored)}" for c in "FBI"
            },
        },  # fmt: skip
        "cat_detected_on_cats": f"{sum(o['is_cat'] for o in cats)}/{len(cats)}",
        "humans_called_cat": f"{sum(o['is_cat'] for o in out if o['label'] == 'human')}/{sum(o['label'] == 'human' for o in out)}",
        "predicted_stressed_share": float(np.mean([o["results"][0][0] == "I" for o in out if o["has_meow"]])),
        "purr": detector("Purr", ("purr",)),
        "hiss": detector("Hiss", ("hiss",)),
        "yowl": detector("Caterwaul", ("yowl", "heat"), among=("hiss",)),  # fights and mating calls; hisses often come with a yowl
        "clips_detail": out,
    }
    Path(__file__).with_name("wild_report.json" if a.encoder == "beats" else f"wild_report_{a.encoder}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False)
    )
    print(json.dumps({k: v for k, v in report.items() if k != "clips_detail"}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--source", choices=["commons", "freesound"], default="commons")
    sub.add_parser("download")
    e = sub.add_parser("evaluate")
    e.add_argument("--encoder", default="beats")
    a = ap.parse_args()
    {"search": search, "download": download, "evaluate": evaluate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
