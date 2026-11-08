#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "build_c_residual_dataset.py" >/dev/null 2>&1; do
  echo "waiting for C residual builder..."
  sleep 30
done
.venv/bin/python scripts/combine_residual_datasets.py
.venv/bin/python scripts/augment_residual_negatives.py
.venv/bin/python scripts/train_residual.py \
  --train artifacts/dataset/residual_v6_aug/train.jsonl \
  --validation artifacts/dataset/residual_v6_aug/validation.jsonl \
  --require-c \
  --dry-run
echo "READY: run the training command from RESIDUAL_MODEL.md"
