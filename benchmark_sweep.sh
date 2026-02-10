#!/bin/bash

MODEL="./Llama-3.2-1B"  # which model
GPU_ID=3                # which GPU
GPU_NAME="v100"         # which GPU
OUTDIR="results/single-gpu/${GPU_NAME}"

mkdir -p "$OUTDIR" # make output directory

run_bench() {
  ISL=$1
  OSL=$2
  BATCH=$3
  TAG=$4
  
  FILENAME="${OUTDIR}/run_isl${ISL}_osl${OSL}_batch${BATCH}.json"
  
  echo "running: ISL=$ISL, OSL=$OSL, batch=$BATCH, tag=$TAG"
  
  CUDA_VISIBLE_DEVICES=$GPU_ID vllm bench throughput \
    --model $MODEL \
    --input-len $ISL \
    --output-len $OSL \
    --num-prompts 500 \
    --max-num-seqs $BATCH \
    --gpu-memory-utilization 0.85 \
    --output-json $FILENAME
  
  echo "completed: $FILENAME"
  echo "---"
}

echo "starting benchmark sweep on GPU: $GPU_NAME"
echo "model: $MODEL"
echo "output directory: $OUTDIR"
echo "========================================"

# prefill sweep (vary ISL, fixed OSL)
echo "PREFILL SWEEP"
run_bench 128  128 1 "short-prefill"
run_bench 512  128 1 "medium-prefill"
run_bench 2048 128 1 "long-prefill"

# decode sweep (fixed ISL, vary OSL)
echo "DECODE SWEEP"
run_bench 512 256  1 "medium-decode"
run_bench 512 512  1 "long-decode"
run_bench 512 1024 1 "extreme-decode"

# heavy workload (long ISL + long OSL)
echo "HEAVY WORKLOAD"
run_bench 2048 512 1 "heavy"

# batching effect (fixed ISL/OSL, vary batch)
echo "BATCHING SWEEP"
run_bench 512 128 8  "batch-8"
run_bench 512 128 16 "batch-16"

echo "========================================"
echo "benchmark completed!"
echo "results saved to: $OUTDIR"