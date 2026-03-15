#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"

python3 -m pip install -r requirements.txt
mkdir -p artifacts artifacts/plots

if [[ "$MODE" == "quick" ]]; then
  python3 scripts/exp_gradient_suppression.py \
    --model gpt2 \
    --max-batches 8 \
    --batch-size 2 \
    --seq-len 64 \
    --sample-tokens-per-batch 64 \
    --output artifacts/gradient_suppression.json

  python3 scripts/exp_spamlang_bottleneck.py \
    --vocab-sizes 256 512 \
    --hidden-dim 64 \
    --steps 250 \
    --batch-size 64 \
    --output artifacts/spamlang_bottleneck.json \
    --plot artifacts/plots/spamlang_bottleneck.png
else
  python3 scripts/exp_gradient_suppression.py \
    --model gpt2 \
    --max-batches 24 \
    --batch-size 4 \
    --seq-len 96 \
    --sample-tokens-per-batch 96 \
    --output artifacts/gradient_suppression.json

  python3 scripts/exp_spamlang_bottleneck.py \
    --vocab-sizes 256 512 1024 \
    --hidden-dim 96 \
    --steps 700 \
    --batch-size 128 \
    --output artifacts/spamlang_bottleneck.json \
    --plot artifacts/plots/spamlang_bottleneck.png
fi

echo "Reproduction completed in mode=$MODE"
