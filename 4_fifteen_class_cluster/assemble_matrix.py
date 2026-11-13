"""Build the 1,500,000 x 384 predictor matrix in the row order of sample_15class.parquet.

Focal rows come from embeddings_v2.npz (row emb_row), pool rows from X_pool.npy.
Output: data/X_15class.npy, uncompressed so the training jobs can memory-map it.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--root", default="/scratch1/fanjiaqi/game_reviews")
args = p.parse_args()
DATA = Path(args.root) / "data"

meta = pd.read_parquet(DATA / "sample_15class.parquet", columns=["cls", "emb_row"])
n = len(meta)
X = np.empty((n, 384), dtype=np.float32)

pool_rows = np.load(DATA / "X_pool_rows.npy")
X[pool_rows] = np.load(DATA / "X_pool.npy")

focal_rows = np.flatnonzero(meta["cls"].str.startswith("F").to_numpy())
emb_row = meta.loc[focal_rows, "emb_row"].to_numpy()
assert (emb_row >= 0).all() and len(focal_rows) + len(pool_rows) == n
t0 = time.time()
big = np.load(DATA / "embeddings_v2.npz")["X"]
print(f"loaded embeddings_v2.npz {big.shape} in {time.time() - t0:.0f}s", flush=True)
assert big.shape == (4_623_482, 384)
X[focal_rows] = big[emb_row]
del big

assert np.isfinite(X).all()
np.save(DATA / "X_15class.npy", X)
print(f"wrote X_15class.npy {X.shape}", flush=True)
