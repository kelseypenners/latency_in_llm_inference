#!/bin/bash

# server should already be running on port 8000!
# CUDA_VISIBLE_DEVICES=3 vllm serve "./Llama-3.2-1B" --port 8000

MODEL="./Llama-3.2-1B"
OUTDIR="results/single-gpu/v100/"
mkdir -p $OUTDIR

run_serve_bench() {
  ISL=$1
  OSL=$2
  TAG=$3
  
  FILENAME="${OUTDIR}/run_isl${ISL}_osl${OSL}_serve.txt"
  
  echo "========================================"
  echo "running: ISL=$ISL, OSL=$OSL ($TAG)"
  echo "========================================"
  
  CUDA_VISIBLE_DEVICES=3 vllm bench serve \
    --backend openai \
    --model $MODEL \
    --dataset-name random \
    --random-input-len $ISL \
    --random-output-len $OSL \
    --num-prompts 100 \
    2>&1 | tee $FILENAME
  
  echo "completed: $FILENAME"
  echo ""
}

echo "Starting benchmark suite"
echo ""

# prefill sweep (vary ISL, fixed OSL)
run_serve_bench 128  128  "short-prefill"
run_serve_bench 512  128  "medium-prefill"
run_serve_bench 2048 128  "long-prefill"

# decode sweep (fixed ISL, vary OSL)
run_serve_bench 512  256  "medium-decode"
run_serve_bench 512  512  "long-decode"
run_serve_bench 512  1024 "extreme-decode"

# heavy workload
run_serve_bench 2048 512  "heavy"

echo "========================================"
echo "Benchmark suite completed!"
echo "Results saved to: $OUTDIR"