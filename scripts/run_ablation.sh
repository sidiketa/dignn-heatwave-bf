#!/usr/bin/env bash
# Ablation study: 6 (C_IN, C_OUT) configurations of DI-GNN.
# Run the full preprocessing once before: python -m dignn.preprocess
set -euo pipefail

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

CONFIGS=("10 1" "10 3" "10 5" "10 7" "7 3" "14 3")

for CONFIG in "${CONFIGS[@]}"; do
  read -r C_IN C_OUT <<< "$CONFIG"
  NAME="CIN${C_IN}_COUT${C_OUT}"
  echo "=== ${NAME}"

  python -m dignn.preprocess --sequences_only --c_in "$C_IN" --c_out "$C_OUT" \
    2>&1 | tee "$LOG_DIR/preproc_${NAME}.log"

  python -m dignn.train --model dignn --c_in "$C_IN" --c_out "$C_OUT" \
    --loss combined --epochs 100 \
    2>&1 | tee "$LOG_DIR/train_${NAME}.log"
done

echo "Ablation done. Next: python -m dignn.evaluate --run_all"
