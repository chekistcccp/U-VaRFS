#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export KERAS_BACKEND=torch
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/default.yaml}"
mkdir -p experiment/{data,models,cache} results logs

"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3,11): raise SystemExit("Python >= 3.11 is required by current KerasHub.")
PY

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON" -m pip install -q -r requirements.txt
fi

# Optional GPU FAISS auto-install. Failure safely falls back to faiss-cpu.
FAISS_MODE="${INSTALL_FAISS_GPU:-auto}"
if [[ "$FAISS_MODE" != "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_MAJOR="$($PYTHON - <<'PY'
try:
 import torch
 print((torch.version.cuda or '').split('.')[0])
except Exception: print('')
PY
)"
  if [[ "$FAISS_MODE" == "1" || "$CUDA_MAJOR" == "12" ]]; then
    echo "[setup] trying CUDA 12 FAISS GPU wheel"
    "$PYTHON" -m pip uninstall -y -q faiss-cpu >/dev/null 2>&1 || true
    "$PYTHON" -m pip install -q faiss-gpu-cu12 || { echo "[warn] GPU FAISS unavailable; restoring CPU FAISS"; "$PYTHON" -m pip install -q faiss-cpu; }
  fi
fi

"$PYTHON" scripts/download_assets.py --root experiment
"$PYTHON" scripts/run_all.py --config "$CONFIG" 2>&1 | tee logs/run_all.log

echo "Done. Aggregated metrics: results/all_metrics.csv"
