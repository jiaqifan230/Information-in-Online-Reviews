"""Fit the 15-class classifier for one validation fold and save out-of-fold probabilities.

For fold k the training set is every sampled review whose fold is not k: all focal
reviews outside fold k and every review of the pool games assigned to other folds.
The test set is fold k. Two models are fitted on the training set:

  main      penalised multinomial logistic regression, 384 inputs -> 15 classes,
            every class given the same total weight;
  pool5     the auxiliary 5-class model of the proximity section, fitted on pool
            reviews only, used to allocate focal reviews among G1-G5.

The ridge strength C is chosen on an inner split that keeps the outer design: the
inner validation set is the training fold (k+1) mod 5, which holds out whole games
for the pool classes. Standardisation is fitted on the inner-training or training
rows only. Selection uses the class-balanced log loss (mean within class, then across
classes), the same aggregation the evaluation uses.

Output: results/oof_fold{k}.npz with the test row positions, the 15-class and the
5-class probabilities; results/fold{k}_summary.json with the tuning path and timings.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

N_CLASSES, N_POOL, N_FOLDS = 15, 5, 5

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, required=True)
p.add_argument("--root", default="/scratch1/fanjiaqi/game_reviews")
p.add_argument("--grid", default="0.1,1,10,100", help="C values (inverse ridge strength) to try")
p.add_argument("--max-iter", type=int, default=500)
args = p.parse_args()
ROOT = Path(args.root)
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
k = args.fold
grid = [float(c) for c in args.grid.split(",")]


def equal_class_weights(y, n_classes):
    """Weights that give every class the same total weight, summing to len(y)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return (len(y) / n_classes) / counts[y]


def balanced_log_loss(y, P):
    """Mean within class of -log p(true class), then mean across classes; in bits."""
    eps = 1e-12
    nll = -np.log2(np.clip(P[np.arange(len(y)), y], eps, None))
    return float(np.mean([nll[y == c].mean() for c in np.unique(y)]))


def balanced_accuracy(y, P):
    pred = P.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def fit(X, y, C, n_classes, max_iter):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", tol=1e-5)
    clf.fit(scaler.transform(X), y, sample_weight=equal_class_weights(y, n_classes))
    return scaler, clf


def predict(scaler, clf, X):
    return clf.predict_proba(scaler.transform(X)).astype(np.float32)


def tune_and_fit(X, y, is_inner_val, n_classes, label, summary):
    """Choose C on the inner split, refit on all rows, return the fitted pair."""
    path = []
    for C in grid:
        t0 = time.time()
        sc, clf = fit(X[~is_inner_val], y[~is_inner_val], C, n_classes, args.max_iter)
        P = predict(sc, clf, X[is_inner_val])
        rec = dict(C=C, inner_log_loss_bits=balanced_log_loss(y[is_inner_val], P),
                   inner_balanced_accuracy=balanced_accuracy(y[is_inner_val], P),
                   n_iter=int(np.max(clf.n_iter_)), seconds=round(time.time() - t0))
        path.append(rec)
        print(f"[{label}] C={C:g}: inner loss {rec['inner_log_loss_bits']:.4f} bits, "
              f"acc {rec['inner_balanced_accuracy']:.4f}, {rec['n_iter']} iters, {rec['seconds']}s", flush=True)
    best = min(path, key=lambda r: r["inner_log_loss_bits"])["C"]
    t0 = time.time()
    sc, clf = fit(X, y, best, n_classes, args.max_iter)
    summary[label] = dict(grid=path, chosen_C=best, final_n_iter=int(np.max(clf.n_iter_)),
                          final_seconds=round(time.time() - t0),
                          n_train_per_class=np.bincount(y, minlength=n_classes).tolist())
    print(f"[{label}] chosen C={best:g}; final fit {summary[label]['final_seconds']}s", flush=True)
    return sc, clf


meta = pd.read_parquet(ROOT / "data" / "sample_15class.parquet", columns=["cls", "class_idx", "fold"])
X_all = np.load(ROOT / "data" / "X_15class.npy", mmap_mode="r")
assert X_all.shape == (len(meta), 384)
y_all = meta["class_idx"].to_numpy()
fold = meta["fold"].to_numpy()
is_pool = meta["cls"].str.startswith("G").to_numpy()

train = np.flatnonzero(fold != k)
test = np.flatnonzero(fold == k)
t0 = time.time()
X_tr = np.ascontiguousarray(X_all[train])
X_te = np.ascontiguousarray(X_all[test])
print(f"fold {k}: train {len(train):,}, test {len(test):,}; matrices loaded in {time.time() - t0:.0f}s", flush=True)
y_tr, y_te = y_all[train], y_all[test]
inner_val = fold[train] == (k + 1) % N_FOLDS
summary = dict(fold=k, n_train=int(len(train)), n_test=int(len(test)), grid=grid)

sc, clf = tune_and_fit(X_tr, y_tr, inner_val, N_CLASSES, "main", summary)
P15 = predict(sc, clf, X_te)
summary["main"]["test_log_loss_bits"] = balanced_log_loss(y_te, P15)
summary["main"]["test_balanced_accuracy"] = balanced_accuracy(y_te, P15)
print(f"[main] test: loss {summary['main']['test_log_loss_bits']:.4f} bits, "
      f"balanced accuracy {summary['main']['test_balanced_accuracy']:.4f}", flush=True)

pool_tr = is_pool[train]
sc5, clf5 = tune_and_fit(X_tr[pool_tr], y_tr[pool_tr] - 10, inner_val[pool_tr], N_POOL, "pool5", summary)
P5 = predict(sc5, clf5, X_te)
pool_te = is_pool[test]
summary["pool5"]["test_log_loss_bits_pool_rows"] = balanced_log_loss(y_te[pool_te] - 10, P5[pool_te])
summary["pool5"]["test_balanced_accuracy_pool_rows"] = balanced_accuracy(y_te[pool_te] - 10, P5[pool_te])

np.savez(RES / f"oof_fold{k}.npz", rows=test, proba15=P15, proba_pool5=P5)
with open(RES / f"fold{k}_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"saved oof_fold{k}.npz and fold{k}_summary.json", flush=True)
