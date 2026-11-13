#!/bin/bash
# One-time environment setup on the CARC cluster. Run on a login node:
#   bash cluster/setup_env.sh
# Creates a conda environment under scratch and downloads the encoder weights
# there, so compute nodes (which may lack internet access) find them in the cache.
set -euo pipefail

ROOT=/scratch1/fanjiaqi/game_reviews
ENV=$ROOT/env
export HF_HOME=$ROOT/hf_cache
export PIP_CACHE_DIR=$ROOT/pip_cache
# login nodes cap the number of threads a user may start; keep BLAS single-threaded for the check below
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

source /etc/profile.d/modules.sh
module purge
module load conda
eval "$(conda shell.bash hook)"

if [ ! -d "$ENV" ]; then
    conda create -y -p "$ENV" python=3.11 pip
fi
conda activate "$ENV"
pip install --upgrade pip
pip install torch sentence-transformers scikit-learn pandas pyarrow numpy scipy

python - <<'EOF'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("encoder cached:", m.get_sentence_embedding_dimension(), "dims, max_seq_length", m.max_seq_length)
import sklearn, torch
print("sklearn", sklearn.__version__, "torch", torch.__version__, "cuda available on this node:", torch.cuda.is_available())
EOF
echo "environment ready at $ENV"
