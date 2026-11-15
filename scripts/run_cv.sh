#!/usr/bin/env bash
# Train + evaluate one config on every fold, then summarise.
#   scripts/run_cv.sh configs/r3d_18.yaml [extra --set overrides...]
# Env:
#   PYTHON    interpreter command (default "uv run python"; use "python" on Colab)
#   RUNS_DIR  parent directory for results (default runs)
#   CV_DIR    resume into an existing runs/cv_* directory; folds with test results are skipped
set -euo pipefail
export PYTHONUNBUFFERED=1   # stream progress into log files
CONFIG=${1:?config path}; shift || true
NAME=$(basename "$CONFIG" .yaml)
PY=${PYTHON:-uv run python}
CV_DIR=${CV_DIR:-${RUNS_DIR:-runs}/cv_${NAME}_$(date +%Y%m%d-%H%M%S)}
mkdir -p "$CV_DIR"
echo "results -> $CV_DIR"
N_FOLDS=$($PY -c "import json; print(json.load(open('data/splits.json'))['meta']['n_folds'])")
for k in $(seq 0 $((N_FOLDS - 1))); do
  if ls "$CV_DIR"/*_fold${k}_*/test_metrics.json >/dev/null 2>&1; then
    echo "===== fold $k: already done, skipping ====="; continue
  fi
  echo "===== fold $k ====="
  rm -rf "$CV_DIR"/*_fold${k}_*   # discard an interrupted attempt
  $PY scripts/train.py --config "$CONFIG" --fold "$k" --set output_dir="$CV_DIR" "$@"
  RUN=$(ls -dt "$CV_DIR"/*_fold${k}_* | head -1)
  $PY scripts/evaluate.py "$RUN/best.pt" > /dev/null
  rm -f "$RUN/last.pt"   # keep only the selected checkpoint (~130 MB each)
  echo "fold $k test: $($PY -c "import json; v=json.load(open('$RUN/test_metrics.json'))['video']; print(f\"video acc {v['accuracy']:.3f} bal-acc {v['balanced_accuracy']:.3f} auc {v['auc']:.3f}\")")"
done
$PY scripts/summarize_cv.py "$CV_DIR"
