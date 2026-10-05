#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p data/archives data/_extracted data/processed/BMAD

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export KERAS_BACKEND=torch
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/default.yaml}"
DATASETS="${DATASETS:-all}"
WORKERS="${PREPROCESS_WORKERS:-8}"

# The user owns the environment. This script deliberately never pip/conda installs anything.
"$PYTHON" scripts/check_env.py

# Fail early on archive/data problems before downloading a large model.
if [[ "${SKIP_PREPROCESS:-0}" != "1" ]]; then
  PREP=("$PYTHON" scripts/prepare_bmad.py
        --data-root data
        --out-root data/processed/BMAD
        --metadata-root metadata
        --datasets "$DATASETS"
        --workers "$WORKERS")
  [[ "${FORCE_PREPROCESS:-0}" == "1" ]] && PREP+=(--force)
  "${PREP[@]}"
fi

if [[ "${PREPROCESS_ONLY:-0}" == "1" ]]; then
  echo "Preprocessing complete. PREPROCESS_ONLY=1, stopping before model download and experiments."
  exit 0
fi

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  "$PYTHON" scripts/download_model.py --out models/dinov3_vitsplus
fi

mkdir -p results logs
"$PYTHON" scripts/run_all.py --config "$CONFIG" --dataset "$DATASETS" 2>&1 | tee logs/run_all.log

echo "Done. Main tables: results/all_metrics.csv and results/summary_metrics.csv"
