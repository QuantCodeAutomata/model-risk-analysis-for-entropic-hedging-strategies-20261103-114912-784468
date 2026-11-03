#!/usr/bin/env bash
# Run all experiments and tests for this replication.
set -eu

python -m pytest -m "not market_data" -q
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4

echo "All experiments complete. Results are in results/."
