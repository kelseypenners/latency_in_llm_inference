#!/bin/bash

MODEL="../Llama-3.1-8B"
GPU_ID=7
GPU_TYPE="a100"
EXPERIMENT="dataset-comparison"
OUTDIR="/mnt/kpenners/results/single-gpu/${EXPERIMENT}/${GPU_TYPE}/"

mkdir -p $OUTDIR

echo "starting benchmark sweep"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID vllm bench sweep serve \
    --serve-cmd "vllm serve ${MODEL} --port 8000 --gpu-memory-utilization 0.85" \
    --bench-cmd "vllm bench serve --backend openai --model ${MODEL} --save-result --save-detailed --num-warmups 10 --ignore-eos" \
    --num-runs 10 \
    --bench-params "./bench_dataset_params.json" \
    --output-dir $OUTDIR

echo "========================================"
echo "benchmark suite completed!"
echo "results saved to: $OUTDIR"
