# Information in game reviews: code

How much does the text of a Steam review tell us about which game, or which kind of game, it
was written for? A frozen sentence encoder (`bge-small-en-v1.5`) turns each review into a
384-dimensional vector, a cross-fitted multinomial logistic regression predicts the label from
the vector, and the held-out log-loss is read as information in bits. This repository holds the
code that produces every number, table and figure, and the description needed to run it.

Three analyses share the same data and method:

| analysis | classes | sample | code | output folder |
|---|---|---|---|---|
| ten games | 10 focal games | 10,000 reviews per game, 2013–2024 | `2_ten_game_analysis/information_in_reviews.ipynb` | `results/` |
| ten games, robustness | same | 100,000 per game | `2_ten_game_analysis/information_in_reviews_100k.ipynb` | `results_100k/` |
| fifteen classes | 10 focal games + 5 genre pools of 90 further games | 100,000 per class, 2015–2024 | folders 3, 4 and 5 | `results/design15/` |

## Layout

One folder per step, numbered in the order in which the steps run.

```
1_embed_focal_reviews/      embed_focal.py: encodes every review of the ten focal games
2_ten_game_analysis/        the ten-game notebook and its copy with 100,000 reviews per game
3_fifteen_class_sample/     notebook: eligibility rules, the 1,500,000-review sample, validation folds
4_fifteen_class_cluster/    Slurm jobs: encode the pool reviews, fit the five folds
5_fifteen_class_analysis/   notebook: every fifteen-class table and figure
README.md                   this file
DATA.md                     the data files the code expects, their place and their columns
requirements.txt            package versions of the laptop runs
run_notebooks.sh            executes the notebooks of steps 2, 3 and 5
```

The repository holds code and descriptions only. The review data are not included, and neither
are the outputs: the code writes them to `results/` and `results_100k/`, which are created on
the first run. The notebooks are stored with the outputs of their last run. All paths in the
code are relative to the repository root.

## Reproducing

Python 3.12 with the packages in `requirements.txt`. Seeds are fixed in the code.

```bash
pip install -r requirements.txt
```

**0. Data.** Put the review files and the game metadata at the paths listed in `DATA.md`.

**1. Embeddings of the focal reviews.** About 2.5 hours on a laptop CPU.

```bash
python 1_embed_focal_reviews/embed_focal.py              # writes data/embeddings_v2.npz
python 1_embed_focal_reviews/embed_focal.py --check 500  # with the matrix in place: re-encode 500 random rows and compare
```

**2. Ten-game analysis.** `information_in_reviews.ipynb` writes `results/`.

```bash
./run_notebooks.sh ten
```

The robustness run is the same notebook with 100,000 reviews per game. It is optional, writes
`results_100k/` and takes about six hours.

```bash
./run_notebooks.sh 100k
```

**3. Sample for the fifteen-class design.** `design_15class_sampling.ipynb` applies the
eligibility rules to all 100 games, draws the 1,500,000-review sample, assigns the validation
folds and writes the audit tables and `results/design15/sample_15class.parquet`. It reads one
table written by step 2.

```bash
./run_notebooks.sh sampling
```

Steps 2 and 3 together take about three hours on a 12-core laptop with 24 GB of memory, most
of it in step 2.

**4. Fifteen-class predictions on the cluster.** Copy the files of `4_fifteen_class_cluster/`
into a folder `cluster/` on the cluster, together with `sample_15class.parquet` and
`embeddings_v2.npz`, then

```bash
bash cluster/setup_env.sh      # once: environment and encoder weights
sbatch cluster/embed.slurm     # encode 500,000 pool reviews on one GPU, assemble the matrix
sbatch cluster/train.slurm     # five folds as an array job
```

and copy `results/oof_fold*.npz` and `fold*_summary.json` back into `results/design15/cluster/`.
Details and the site-specific settings are in `4_fifteen_class_cluster/README.md`.

**5. Fifteen-class analysis.** `analysis_15class.ipynb` reads the sample and the out-of-fold
predictions and writes every fifteen-class table and figure to `results/design15/`. A few
minutes.

```bash
./run_notebooks.sh analysis
```

## What was checked

The laptop part was re-executed on 11 November 2026 in a copy of this folder with the data in
place, starting from the raw review files, and its outputs were compared with those of the
earlier runs.

| step | code | re-executed | outcome |
|---|---|---|---|
| 1 | `embed_focal.py --check 500` | yes | 500 random rows re-encoded; largest absolute difference from the stored matrix 3.7e-07, cosine similarity 1.000000 |
| 2 | `information_in_reviews.ipynb` | yes, 27 code cells, no errors | every file it writes to `results/` is byte-identical to the earlier run |
| 2 | `information_in_reviews_100k.ipynb` | no | its stored outputs are from the run of 13 October 2026 |
| 3 | `design_15class_sampling.ipynb` | yes, 15 code cells, no errors | the 1,500,000-review sample, the folds and all audit tables are identical to the earlier run |
| 4 | pool embeddings, five folds | no | the fifteen-class analysis used the predictions of the cluster run of October 2026 |
| 5 | `analysis_15class.ipynb` | yes, 19 code cells, no errors | every table and figure in `results/design15/` is byte-identical to the earlier run |

In total 77 output files were regenerated and compared with the earlier ones; none differed.
The notebooks are stored with the outputs of these runs, so a new run can be compared with
them cell by cell.

## Caveats

- The markdown commentary inside `information_in_reviews_100k.ipynb` is copied from the
  10,000-per-game notebook and still quotes its numbers; the cell outputs are the
  100,000-per-game results.
- Results are reproducible to the printed precision, not bit for bit across machines: the
  encoder and the optimiser depend on the linear-algebra library and on batch composition at
  the level of 1e-6.
