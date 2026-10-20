#!/usr/bin/env bash
# Train DI-GNN and the baseline GNN with 10 seeds on the reference configuration CIN10_COUT3.
set -euo pipefail

for SEED in $(seq 42 51); do
  python -m dignn.train --model dignn    --c_in 10 --c_out 3 --loss combined --seed "$SEED"
  python -m dignn.train --model baseline --c_in 10 --c_out 3 --loss standard --seed "$SEED"
done

python -m dignn.evaluate --run_all
