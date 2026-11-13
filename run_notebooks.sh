#!/bin/bash
# executes the notebooks in order: ./run_notebooks.sh [ten] [sampling] [analysis] [100k]
# without arguments: ten, sampling, analysis; the 100,000-per-game run (about 6 hours) only when named
set -euo pipefail
cd "$(dirname "$0")"
run() { (cd "$1" && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$2"); }
[ $# -eq 0 ] && set -- ten sampling analysis
for stage in "$@"; do
    case "$stage" in
        ten)      run 2_ten_game_analysis information_in_reviews.ipynb ;;
        sampling) run 3_fifteen_class_sample design_15class_sampling.ipynb ;;
        analysis)
            if [ -e results/design15/cluster/oof_fold4.npz ]; then
                run 5_fifteen_class_analysis analysis_15class.ipynb
            else
                echo "analysis not run: results/design15/cluster/ holds no out-of-fold predictions (see 4_fifteen_class_cluster/README.md)"
            fi ;;
        100k)     run 2_ten_game_analysis information_in_reviews_100k.ipynb ;;
        *)        echo "unknown stage: $stage"; exit 1 ;;
    esac
done
