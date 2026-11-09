#!/bin/bash
#SBATCH -J selaur-case
#SBATCH -A YOUR_ACCOUNT          # set your Slurm account
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o logs/case_study_%j.out

# Case study: export step-150 checkpoint to HF, then trace full ALFWorld episodes
# for the base model and the trained model on the same validation games.
set -eo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate selaur
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
cd $REPO
CKPT=${CKPT:?set CKPT to <checkpoint_dir>/global_step_150/actor}
HF=${HF:-${CKPT_ROOT:-checkpoints}/hf/alfworld_selaur_step150}
OUT=$REPO/logs/case_study

if [ ! -f $HF/config.json ]; then
  echo "=== merge checkpoint -> $HF ==="
  python -m verl.model_merger merge --backend fsdp --local_dir $CKPT --target_dir $HF
fi
ls -la $HF

FILTERS="pick_cool_then_place_in_recep-Apple-None-CounterTop pick_heat_then_place_in_recep pick_and_place_simple pick_clean_then_place_in_recep"
echo "=== trace: trained (step 150) ==="
python -W ignore scripts/case_study_trace.py --model $HF --tag trained --games-filter $FILTERS --max-games 2 --out-dir $OUT
echo "=== trace: base (Qwen2.5-1.5B-Instruct) ==="
python -W ignore scripts/case_study_trace.py --model Qwen/Qwen2.5-1.5B-Instruct --tag base --games-filter $FILTERS --max-games 2 --out-dir $OUT
echo "=== DONE ==="
