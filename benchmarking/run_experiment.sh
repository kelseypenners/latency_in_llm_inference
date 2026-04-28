#!/bin/bash
# run_experiment.sh
# usage: ./run_experiment.sh <config.json>

set -e

CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "usage: ./run_experiment.sh <config.json>"
    exit 1
fi

# read config file
experiment=$(jq -r '.experiment' "$CONFIG")
experiment_type=$(jq -r '.experiment_type' "$CONFIG")
label=$(jq -r '.label // ""' "$CONFIG")
model=$(jq -r '.model' "$CONFIG")
gpu_ids=$(jq -r '.gpu_ids' "$CONFIG")
gpu_type=$(jq -r '.gpu_type' "$CONFIG")
tp=$(jq -r '.tp // 1' "$CONFIG")
runs=$(jq -r '.runs // 1' "$CONFIG")
gpu_mem=$(jq -r '.gpu_mem // 0.85' "$CONFIG")
serve_extra=$(jq -r '.serve_cmd_extra // ""' "$CONFIG")
bench_extra=$(jq -r '.bench_cmd_extra' "$CONFIG")
base_outdir=$(jq -r '.results' "$CONFIG")
disable_p2p=$(jq -r '.disable_p2p // ""' "$CONFIG")
network_fallback=$(jq -r '.network_fallback // ""' "$CONFIG")
dry_run=$(jq -r '.dry_run // ""' $CONFIG)
resume=$(jq -r '.resume // ""' $CONFIG)
nccl_debug=$(jq -r '.nccl_debug // ""' "$CONFIG")
outdir="${base_outdir}/${experiment_type}/${experiment}/${gpu_type}/"
mkdir -p "$outdir"

# write params files
bench_params_flag=""
serve_params_flag=""
if jq -e '.bench_params' "$CONFIG" > /dev/null 2>&1; then
    bench_params_file=$(mktemp /tmp/bench_params_XXXX.json)
    jq '.bench_params' "$CONFIG" > "$bench_params_file"
    bench_params_flag="--bench-params $bench_params_file"
fi
if jq -e '.serve_params' "$CONFIG" > /dev/null 2>&1; then
    serve_params_file=$(mktemp /tmp/serve_params_XXXX.json)
    jq '.serve_params' "$CONFIG" > "$serve_params_file"
    serve_params_flag="--serve-params $serve_params_file"
fi

topology=$(nvidia-smi topo -m)

# NCCL env vars for interconnect control
nccl_env=()
nccl_comm="default"

if [ "$disable_p2p" = "true" ]; then
    # disable ussing NCCL_P2P_DISABLE
    nccl_env=(NCCL_P2P_DISABLE=1)
    nccl_comm="P2P disabled" 
fi
if [ "$network_fallback" = "true" ]; then
    # disable ussing NCCL_P2P_DISABLE and NCCL_SHM_DISABLE
    nccl_env=(NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1)
    nccl_comm="network fallback, (P2P, SHM disabled)"
fi
if [ -n "$nccl_debug" ]; then
    # enable NCCL_DEBUG
    nccl_env+=(NCCL_DEBUG=$nccl_debug)
    nccl_env+=(NCCL_DEBUG_FILE="$outdir/nccl_debug_%h_%p.log")
fi

# label flag
label_flag=""
if [ "$label" != "" ]; then
    label_flag="--label $label"
fi

# dry run flag
dry_run_flag=""
if [ "$dry_run" = "true" ]; then
    dry_run_flag="--dry-run"
fi

# resume flag
resume_flag=""
if [ "$resume" != "" ]; then
    resume_flag="--resume $resume"
fi

# GPU power usage logging
power_log="${outdir}/gpu_power.csv"

# log every 5 seconds
nvidia-smi \
  --query-gpu=timestamp,index,power.draw \
  --format=csv,noheader,nounits \
  -l 5 > "$power_log" &

POWER_LOG_PID=$!

# ensure logger is killed even if script exits early
cleanup() {
    rm -f "${bench_params_file:-}" "${serve_params_file:-}"
    if ps -p $POWER_LOG_PID > /dev/null 2>&1; then
        kill $POWER_LOG_PID
        wait $POWER_LOG_PID 2>/dev/null
        echo "power logging stopped"
    fi
}
trap cleanup EXIT

echo "================================================================================"
echo "experiment    : $experiment ($experiment_type) $label"
echo "model         : $model"
echo "gpu id(s)     : $gpu_ids  ($gpu_type)"
echo "gpu mem       : $gpu_mem"
if [ "$nccl_comm" != "" ]; then
    echo "communication : $nccl_comm"
fi
if [ -n "$nccl_debug" ]; then
    echo "nccl debug    : $nccl_debug"
fi
echo "tp            : $tp"
echo "runs          : $runs"
echo "output        : $outdir"
if [ "$dry_run" = "true" ]; then
    echo "mode           : DRY RUN"
fi
if [ "$resume" != "" ]; then
    echo "resuming run  : $resume"
fi
echo "================================================================================"
echo ""

# run sweep
env CUDA_VISIBLE_DEVICES=$gpu_ids "${nccl_env[@]}" \
    vllm bench sweep serve \
    --serve-cmd "vllm serve ${model} \
    --tensor-parallel-size ${tp} \
    --gpu-memory-utilization ${gpu_mem} ${serve_extra}" \
    --bench-cmd "vllm bench serve \
    --model ${model} ${label_flag} ${bench_extra}" \
    --num-runs "$runs" \
    $bench_params_flag $serve_params_flag \
    --output-dir "$outdir" \
    $dry_run_flag 

# log experiment
log="${base_outdir}/experiment_log.csv"
date_str=$(date +%d-%m-%Y)
if [ ! -f "$log" ]; then
    echo "date,experiment,experiment_type,gpu_type,model,tp,outdir,config" > "$log"
fi
echo "${date_str},${experiment},${experiment_type}:${label},${gpu_type},${model},${tp},${outdir},${CONFIG}" >> "$log"

echo "========================================"
echo "benchmark suite completed!"
echo "results saved to: $outdir"
echo "experiment log updated: $log"