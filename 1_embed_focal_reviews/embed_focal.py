"""Encode every review of the ten focal games with bge-small-en-v1.5.

Writes data/embeddings_v2.npz, a 4,623,482 x 384 float32 matrix under the key "X".
Row i of the matrix is row i of the concatenation of the ten feather files in the
order of GAME_FILES, all languages, with a missing review encoded as the empty string.
The notebooks rely on this row order. About 2.5 hours on a laptop CPU.

    python 1_embed_focal_reviews/embed_focal.py              # encode everything and write the matrix
    python 1_embed_focal_reviews/embed_focal.py --check 500  # encode 500 random rows and compare
                                                             # them with the existing matrix
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "embeddings_v2.npz"

GAME_FILES = [
    "game_reviews_218620.feather",  # PAYDAY 2
    "game_reviews_236390.feather",  # War Thunder
    "game_reviews_251570.feather",  # 7 Days to Die
    "game_reviews_284160.feather",  # BeamNG.drive
    "game_reviews_304930.feather",  # Unturned
    "game_reviews_377160.feather",  # Fallout 4
    "game_reviews_381210.feather",  # Dead by Daylight
    "game_reviews_413150.feather",  # Stardew Valley
    "game_reviews_438100.feather",  # VRChat
    "game_reviews_444090.feather",  # Paladins
]

p = argparse.ArgumentParser()
p.add_argument("--check", type=int, default=0, metavar="N",
               help="encode N random rows and compare with the existing matrix instead of rewriting it")
args = p.parse_args()

texts = pd.concat([pd.read_feather(RAW / f, columns=["review"]) for f in GAME_FILES],
                  ignore_index=True)["review"].fillna("").to_numpy()
print(f"{len(texts):,} reviews")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

if args.check:
    rows = np.sort(np.random.default_rng(0).choice(len(texts), size=args.check, replace=False))
    new = model.encode(list(texts[rows]), batch_size=32, convert_to_numpy=True, normalize_embeddings=False)
    old = np.load(OUT)["X"]
    assert old.shape == (len(texts), 384), old.shape
    old = old[rows]
    cos = (new * old).sum(1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(old, axis=1))
    print(f"{args.check} rows: largest absolute difference {np.abs(new - old).max():.2e}, "
          f"smallest cosine similarity {cos.min():.6f}")
else:
    X = model.encode(list(texts), batch_size=32, show_progress_bar=True, convert_to_numpy=True,
                     normalize_embeddings=False)
    np.savez_compressed(OUT, X=X.astype(np.float32))
    print(f"wrote {OUT} {X.shape}")
