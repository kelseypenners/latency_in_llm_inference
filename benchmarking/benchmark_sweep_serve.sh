#!/bin/bash

# server should already be running on port 8000!
# CUDA_VISIBLE_DEVICES=3 vllm serve "./Llama-3.2-1B" --port 8000

MODEL="./Llama-3.2-1B"
GPU_ID=3
GPU_TYPE="v100"
DATE="19-02-2026"
OUTDIR="../results/single-gpu/${DATE}/${GPU_TYPE}/"

mkdir -p $OUTDIR

echo "starting benchmark sweep"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID vllm bench sweep serve \
    --serve-cmd "vllm serve ${MODEL} -- port 8000" \
    --bench-cmd "vllm bench serve --backend openai --model ${MODEL} --dataset-name random --num-prompts 100" \
    --num-runs 5 \
    --bench-params "./bench_params.json" \
    --output-dir $OUTDIR

echo "========================================"
echo "benchmark suite completed!"
echo "results saved to: $OUTDIR"