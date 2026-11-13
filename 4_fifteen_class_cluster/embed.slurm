#!/bin/bash
#SBATCH --account=gourab_809
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=embed_pool
#SBATCH --output=/scratch1/fanjiaqi/game_reviews/logs/embed_%j.out

# Step 1 of the cluster pipeline: encode the pool reviews on a GPU, then assemble
# the full 1.5M x 384 matrix (focal vectors copied from embeddings_v2.npz).
set -euo pipefail
ROOT=/scratch1/fanjiaqi/game_reviews
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1

source /etc/profile.d/modules.sh
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate $ROOT/env

nvidia-smi -L
cd $ROOT/cluster
python embed_pool.py --root $ROOT
python assemble_matrix.py --root $ROOT
