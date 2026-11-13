# Cluster step of the 15-class design

Runs on the USC CARC Discovery cluster (Slurm, account `gourab_809`). The laptop notebook
`3_fifteen_class_sample/design_15class_sampling.ipynb` produces `results/design15/sample_15class.parquet`;
this folder turns it into out-of-fold probabilities for every sampled review.

Layout on the cluster, all under `/scratch1/fanjiaqi/game_reviews`:

| path | what |
|---|---|
| `cluster/` | the files of this folder (uploaded) |
| `data/sample_15class.parquet` | the 1.5 M-review sample (uploaded) |
| `data/embeddings_v2.npz` | 4,623,482 x 384 vectors of the ten focal games (uploaded) |
| `data/X_pool.npy`, `data/X_15class.npy` | produced by step 1 |
| `results/oof_fold{k}.npz`, `results/fold{k}_summary.json` | produced by step 2 |
| `env/`, `hf_cache/` | conda environment and encoder weights |
| `logs/` | Slurm output files |

## Commands

```bash
# log in (USC Secure Wireless or USC VPN required)
ssh usc

# one-time: environment and encoder weights (login node, ~10 min)
bash /scratch1/fanjiaqi/game_reviews/cluster/setup_env.sh

# step 1: encode pool reviews on a GPU and assemble the predictor matrix
sbatch /scratch1/fanjiaqi/game_reviews/cluster/embed.slurm

# step 2: five folds in parallel (after step 1 finishes)
sbatch /scratch1/fanjiaqi/game_reviews/cluster/train.slurm

# watch the queue; cancel with scancel <jobid>
squeue -u fanjiaqi
```

Sync from the laptop (run in the repository root):

```bash
rsync -av 4_fifteen_class_cluster/ usc:/scratch1/fanjiaqi/game_reviews/cluster/
rsync -av results/design15/sample_15class.parquet data/embeddings_v2.npz usc:/scratch1/fanjiaqi/game_reviews/data/
rsync -av usc:/scratch1/fanjiaqi/game_reviews/results/ results/design15/cluster/
```

## What the scripts do

- `embed_pool.py` encodes the 500,000 pool reviews with `bge-small-en-v1.5`, the same encoder,
  settings and no normalisation as the original run that produced `embeddings_v2.npz`.
- `assemble_matrix.py` writes the 1,500,000 x 384 matrix in the parquet's row order: focal rows
  copied from `embeddings_v2.npz` by `emb_row`, pool rows from `X_pool.npy`.
- `train_folds.py --fold k` fits the 15-class penalised multinomial logistic regression on every
  review outside fold k with equal total class weight, chooses the ridge strength on an inner
  split that holds out whole pool games, and saves the probabilities for fold k. It also fits the
  auxiliary 5-class pool model used for the proximity measure.

## Site-specific settings

The scripts are kept as they were run. Two things are tied to the USC CARC cluster and need
editing elsewhere: the Slurm account and partition lines (`#SBATCH --account=gourab_809`,
`--partition`, `--constraint`) and the root directory `/scratch1/fanjiaqi/game_reviews`, which
appears in `ROOT=` and in the `#SBATCH --output` lines of the two `.slurm` files and in
`setup_env.sh`. The Python scripts take the root as `--root`.

The GPU constraint excludes V100 and P100 cards, which the installed PyTorch build does not support.

## Versions and run record

Python 3.11, scikit-learn 1.9.1, PyTorch 2.14.1 (CUDA 13), pandas 3.0.6, numpy 2.4.6,
installed by `setup_env.sh` on 5 October 2026. In the run of October 2026,
encoding 500,000 pool reviews took 142 seconds on one GPU and each fold about fifteen minutes
on sixteen cores.
