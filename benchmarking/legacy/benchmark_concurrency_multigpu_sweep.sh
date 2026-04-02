#!/bin/bash

MODEL="../Llama-3.1-8B"
GPU_ID=6,7
GPU_TYPE="a100"
EXPERIMENT="concurrency-sweep"
OUTDIR="../results/multi-gpu/${EXPERIMENT}/${GPU_TYPE}/"

mkdir -p $OUTDIR

echo "starting benchmark sweep"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID vllm bench sweep serve \
    --serve-cmd "vllm serve ${MODEL} --port 8000 --gpu-memory-utilization 0.85 --max-num-seqs 1024 \
    --tensor-parallel-size 2 --max-model-len 80000" \
    --bench-cmd "vllm bench serve --backend openai --model ${MODEL} --dataset-name random --random-input-len 512 \
    --random-output-len 512 --save-result --metric-percentiles 75,90,99 \
    --num-warmups 20 --ignore-eos --ready-check-timeout-sec 2000" \
    --num-runs 2 \
    --bench-params "./bench_concurrency_params.json" \
    --output-dir $OUTDIR\

echo "========================================"
echo "benchmark suite completed!"
echo "results saved to: $OUTDIR"