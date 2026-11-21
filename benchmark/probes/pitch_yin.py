"""Same experiment as pitch.py, with the pitch tracker the package ships (core.pitch_features). -> data/wild/pitch_yin.npz"""
import json, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "benchmark")
from wild import WILD, labelled, local_name
from meow_decoder import bench, core
files, y, cats, _ = bench.catmeows("data/dataset")
C = np.array([core.pitch_features(core.load(f)) for f in files])
rows = labelled()
W = np.array([core.pitch_features(core.decode((WILD / "audio" / local_name(r)).read_bytes())) for r in rows])
np.savez("data/wild/pitch_yin.npz", catmeows=C, y=y, cats=cats, wild=W, meta=json.dumps([{"label": r["label"], "uploader": r["artist"], "title": r["title"]} for r in rows]))
