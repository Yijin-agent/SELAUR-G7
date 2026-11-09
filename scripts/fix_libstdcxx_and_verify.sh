#!/bin/bash
set -eo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate selaur
conda install -y --override-channels -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/libstdcxx.sh" <<'A'
export _SELAUR_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
A
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/libstdcxx.sh" <<'D'
export LD_LIBRARY_PATH="${_SELAUR_OLD_LD_LIBRARY_PATH:-}"; unset _SELAUR_OLD_LD_LIBRARY_PATH
D
conda deactivate; conda activate selaur
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep -q GLIBCXX_3.4.29 && echo "GLIBCXX_3.4.29 present"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
python -W ignore -c "import torch, verl, selaur, vllm, flash_attn, alfworld; print('torch', torch.__version__, '| vllm', vllm.__version__, '| flash_attn', flash_attn.__version__); print('IMPORT OK')"
pip install -q pytest
pytest tests/selaur -q -p no:cacheprovider 2>&1 | tail -15
