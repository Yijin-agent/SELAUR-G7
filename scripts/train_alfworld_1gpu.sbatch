#!/bin/bash
#SBATCH -J selaur-alfworld
#SBATCH -A YOUR_ACCOUNT          # set your Slurm account
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=192G       # smoke test peaked at ~105GB (128 ALFWorld env actors)
#SBATCH -t 3-00:00:00    # ~34h of training + validation; resubmit to resume if it runs out
#SBATCH -o logs/train_alfworld_%j.out

# Full SELAUR run: ALFWorld, Qwen2.5-1.5B-Instruct, 150 steps, 1x A100-80GB.
# Checkpoints every 10 steps to /scratch; resume_mode=auto picks up the latest one on resubmit.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate selaur
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
nvidia-smi --query-gpu=name,memory.total --format=csv
bash examples/selaur_trainer/run_alfworld_1gpu.sh
