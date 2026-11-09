#!/bin/bash
#SBATCH -J selaur-install
#SBATCH -A YOUR_ACCOUNT          # set your Slurm account
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/install_%j.out

# Builds the `selaur` conda env (ALFWorld variant). Run with: sbatch scripts/install_env.sbatch
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
cd "$REPO"

echo "=== [1/6] conda env ==="
if ! conda env list | grep -qE '^selaur\s'; then
    conda create -n selaur python=3.12 -y --override-channels -c conda-forge
fi
conda activate selaur
python --version
pip install -U pip wheel "setuptools>=74.1.1,<81"   # verl needs pkg_resources (removed in setuptools 81+)

echo "=== [2/6] torch ==="
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== [3/6] repo (verl + selaur) ==="
pip install -e .

echo "=== [4/6] vllm ==="
pip install vllm==0.8.5

echo "=== [5/6] flash-attn ==="
# Prebuilt wheel for torch2.6 / cu12 / cp312; avoids compiling from source.
pip install flash-attn==2.7.4.post1 --no-build-isolation || \
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

echo "=== [6/6] ALFWorld ==="
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
# vllm 0.8.5 pins opentelemetry 1.26; keep ray/wandb extras compatible with it
pip install "ray[default]==2.46.0" "opentelemetry-api==1.26.0" "opentelemetry-sdk==1.26.0" "opentelemetry-exporter-prometheus==0.47b0" "protobuf>=3.20,<5" "google-api-core<2.25" "proto-plus<1.26" "googleapis-common-protos<1.70" "wandb<0.20"
alfworld-download -f

echo "=== verify ==="
python -c "import torch, verl, selaur, vllm, flash_attn, alfworld; print('torch', torch.__version__, '| vllm', vllm.__version__, '| flash_attn', flash_attn.__version__); print('ok')"
pip install pytest
pytest tests/selaur -q
echo "=== DONE ==="
