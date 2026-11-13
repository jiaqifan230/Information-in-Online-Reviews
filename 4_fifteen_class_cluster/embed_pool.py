"""Encode the pool (G1-G5) reviews of the 15-class sample with bge-small-en-v1.5.

Focal reviews already have vectors in embeddings_v2.npz (column emb_row of the sample);
only the 500,000 pool reviews are new. The encoder call matches the original run
(_archive/full_pipeline.py): same model, no normalisation, default 512-token window.

Output: data/X_pool.npy (n_pool x 384 float32) and data/X_pool_rows.npy, the row
positions of those reviews in sample_15class.parquet.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

p = argparse.ArgumentParser()
p.add_argument("--root", default="/scratch1/fanjiaqi/game_reviews")
p.add_argument("--batch-size", type=int, default=256)
args = p.parse_args()
DATA = Path(args.root) / "data"

meta = pd.read_parquet(DATA / "sample_15class.parquet", columns=["cls", "emb_row", "text"])
pool_rows = np.flatnonzero(meta["cls"].str.startswith("G").to_numpy())
assert (meta.loc[pool_rows, "emb_row"] == -1).all()
texts = meta.loc[pool_rows, "text"].tolist()
del meta
print(f"{len(texts):,} pool reviews to encode; cuda: {torch.cuda.is_available()}", flush=True)

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()
X = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                 normalize_embeddings=False, show_progress_bar=False)
X = X.astype(np.float32)
assert X.shape == (len(texts), 384)
np.save(DATA / "X_pool.npy", X)
np.save(DATA / "X_pool_rows.npy", pool_rows)
print(f"encoded in {time.time() - t0:.0f}s -> {X.shape}", flush=True)
