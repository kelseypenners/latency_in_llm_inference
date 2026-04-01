#!/bin/bash

MODEL="/mnt/kpenners/Llama-3.1-8B"
GPU_ID=3
GPU_TYPE="v100"
EXPERIMENT="seqlen-sweep"
OUTDIR="/mnt/kpenners/results/single-gpu/${EXPERIMENT}/${GPU_TYPE}/"

mkdir -p $OUTDIR

echo "starting benchmark sweep"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID vllm bench sweep serve \
    --serve-cmd "vllm serve ${MODEL} --port 8000 --gpu-memory-utilization 0.85 --max-model-len 85000" \
    --bench-cmd "vllm bench serve --backend openai --model ${MODEL} --dataset-name random \
    --num-prompts 25 --save-result --num-warmups 20 --ignore-eos --max-concurrency 1" \
    --num-runs 2 \
    --bench-params "./bench_seqlen_params_extended.json" \
    --output-dir $OUTDIR

echo "========================================"
echo "benchmark suite completed!"
echo "results saved to: $OUTDIR"