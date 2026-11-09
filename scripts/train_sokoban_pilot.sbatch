#!/bin/bash
#SBATCH -J selaur-sokoban
#SBATCH -A YOUR_ACCOUNT          # set your Slurm account
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o logs/train_sokoban_%j.out

# SELAUR pilot on Sokoban (text, 6x6, 1 box), Qwen2.5-1.5B, 30 steps, 1x A100.
# Same algorithm hyperparameters as the ALFWorld run; val_before_train gives the base-model baseline.
export PYTHONUNBUFFERED=1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate selaur
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
bash examples/selaur_trainer/run_sokoban_text_1gpu.sh
