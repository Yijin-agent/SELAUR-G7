#!/bin/bash
#SBATCH -J selaur-smoke
#SBATCH -A YOUR_ACCOUNT          # set your Slurm account
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=256G    # 128 ALFWorld env actors (16 tasks x 8 rollouts) need lots of host RAM
#SBATCH -t 4:00:00
#SBATCH -o logs/smoke_%j.out

# 1-GPU end-to-end smoke test: SELAUR on ALFWorld, Qwen2.5-1.5B, 2 epochs.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate selaur
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
nvidia-smi --query-gpu=name,memory.total --format=csv
python -W ignore -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bash examples/selaur_trainer/smoke_alfworld_1gpu.sh
