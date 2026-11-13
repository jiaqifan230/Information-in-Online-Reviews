# Data

The data are not part of the repository (about 10 GB). The code expects them at the paths
below, relative to the repository root; the folders `data/` and `game data 2/` have to be
created there. The review files and the two metadata files are the inputs; the other three
files are produced by the code.

| path | size | what |
|---|---|---|
| `data/raw/game_reviews_<appid>.feather` | 0.8 GB | every Steam review of the ten focal games, all languages: 4,623,482 records, June 2013 to November 2024 |
| `data/game_info.csv` | 0.1 MB | store metadata of the ten focal games |
| `data/embeddings_v2.npz` | 6.5 GB | 4,623,482 x 384 `bge-small-en-v1.5` vectors of the focal reviews, key `X`; produced by `1_embed_focal_reviews/embed_focal.py` |
| `game data 2/90 popular games/game_reviews_<appid>.feather` | 3.0 GB | every Steam review of the 90 pool games: 16,079,097 records |
| `game data 2/popular_games_info.feather` | 0.4 MB | store metadata of the 90 pool games |
| `results/design15/sample_15class.parquet` | 130 MB | the 1,500,000-review sample of the fifteen-class design with text, class, fold and embedding row; written by `3_fifteen_class_sample/design_15class_sampling.ipynb` |
| `results/design15/eligible_meta.parquet` | 115 MB | cache of the eligibility pass over all 20.7 million records; same notebook |

The ten focal games and the 90 pool games are together the 100 games with the most reviews in
the source collection, restricted to games released before 2020.

## Review files

One row per review as returned by the Steam reviews API. Columns used by the analysis:

| column | use |
|---|---|
| `recommendationid` | review identifier; repeated records are removed by it |
| `language` | Steam's language tag; the analysis keeps `english` |
| `review` | review text |
| `timestamp_created` | Unix time of creation; reduced to calendar year |
| `timestamp_updated` | differs from `timestamp_created` when the review was edited |
| `voted_up` | the verdict: recommended or not |
| `votes_up`, `votes_funny`, `weighted_vote_score` | helpfulness and funny votes |
| `appid` | Steam application id (also in the file name) |

The other columns (`author`, `comment_count`, `steam_purchase`, `received_for_free`,
`written_during_early_access`, `developer_response`, ...) are carried along and not used.

## Embedding matrix

Row `i` of `embeddings_v2.npz` is row `i` of the concatenation of the ten focal files in the
order listed in `1_embed_focal_reviews/embed_focal.py` (and in the first cell of the notebooks). The notebooks
rebuild the metadata in that order and assert that the row counts agree. The vectors of the pool
reviews are not stored here: only the 500,000 sampled pool reviews are encoded, on the cluster,
by `4_fifteen_class_cluster/embed_pool.py`.
