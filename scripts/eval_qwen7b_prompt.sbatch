#!/bin/bash
#SBATCH -J qwen7b-eval
#SBATCH -A YOUR_ACCOUNT          # set your Slurm account
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=320G     # 7B actor+optimizer offloaded to CPU + 256 ALFWorld env actors
#SBATCH -t 4:00:00
#SBATCH -o logs/eval_qwen7b_%j.out

# Baseline: Qwen2.5-7B-Instruct, prompt only (no RL), evaluated with val_only on the
# same 128 validation tasks / prompts / memory / action parsing as the SELAUR runs.
export PYTHONUNBUFFERED=1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate selaur
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
echo "=== SOKOBAN ==="
bash examples/baselines/eval_sokoban_qwen7b_prompt.sh || echo "SOKOBAN EVAL FAILED"
ray stop --force >/dev/null 2>&1 || true
echo "=== ALFWORLD ==="
bash examples/baselines/eval_alfworld_qwen7b_prompt.sh || echo "ALFWORLD EVAL FAILED"
echo "=== DONE ==="
