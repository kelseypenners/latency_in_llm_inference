#!/bin/bash

MODEL="/mnt/kpenners/Llama-3.2-1B"
GPU_ID=7
GPU_TYPE="a100"
EXPERIMENT="concurrecy-sweep"
OUTDIR="/mnt/kpenners/results/single-gpu/${EXPERIMENT}/${GPU_TYPE}/"

mkdir -p $OUTDIR

echo "starting benchmark sweep"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID vllm bench sweep serve \
    --serve-cmd "vllm serve ${MODEL} --port 8000 --gpu-memory-utilization 0.85" \
    --bench-cmd "vllm bench serve --backend openai --model ${MODEL} --dataset-name random --random-input-len 512 --random-output-len 512 --num-prompts 500 --save-result --save-detailed --metric-percentiles 75,90,99 --num-warmups 20 --ignore-eos" \
    --num-runs 2 \
    --bench-params "./bench_concurrency_params.json" \
    --output-dir $OUTDIR

echo "========================================"
echo "benchmark suite completed!"
echo "results saved to: $OUTDIR"