#!/bin/bash
#SBATCH --account=gourab_809
#SBATCH --partition=main
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=train15
#SBATCH --output=/scratch1/fanjiaqi/game_reviews/logs/train_fold%a_%j.out

# Step 2 of the cluster pipeline: one array task per validation fold.
# Submit with:  sbatch cluster/train.slurm
set -euo pipefail
ROOT=/scratch1/fanjiaqi/game_reviews
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /etc/profile.d/modules.sh
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate $ROOT/env

cd $ROOT/cluster
python train_folds.py --fold $SLURM_ARRAY_TASK_ID --root $ROOT
