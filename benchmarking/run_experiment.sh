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
model=$(jq -r '.model' "$CONFIG")
gpu_ids=$(jq -r '.gpu_ids' "$CONFIG")
gpu_type=$(jq -r '.gpu_type' "$CONFIG")
tp=$(jq -r '.tp // 1' "$CONFIG")
runs=$(jq -r '.runs // 1' "$CONFIG")
gpu_mem=$(jq -r '.gpu_mem // 0.85' "$CONFIG")
serve_extra=$(jq -r '.serve_cmd_extra' "$CONFIG")
bench_extra=$(jq -r '.bench_cmd_extra' "$CONFIG")
base_outdir=$(jq -r '.results' "$CONFIG")
disable_p2p=$(jq -r '.disable_p2p // ""' "$CONFIG")
network_fallback=$(jq -r '.network_fallback // ""' "$CONFIG")
outdir="${base_outdir}/${experiment_type}/${experiment}/${gpu_type}/"
mkdir -p "$outdir"

# write params to temp files
bench_params_file=$(mktemp /tmp/bench_params_XXXX.json)
serve_params_file=$(mktemp /tmp/serve_params_XXXX.json)
jq '.bench_params' "$CONFIG" > "$bench_params_file"
jq '.serve_params' "$CONFIG" > "$serve_params_file"
trap "rm -f $bench_params_file $serve_params_file" EXIT

topology=$(nvidia-smi topo -m)
if [ "$disable_p2p" != "" ]; then
    # check if nvlink exists
    if [[ $topology == *"NV"*"NV"*"NV" ]]; then
        interconnect= "NVLink"
    else
        interconnect= "PCIe (NVLink not detected)"

    
fi
if [ "$interconnect" != "pcie" ]; then
    if [[ $topology == *"NV"*"NV"*"NV" ]]; then
        interconnect= "PCIe (using )"
    else
        interconnect= "PCIe (NVLink not detected)"
fi

echo "========================================"
echo "experiment   : $experiment ($experiment_type)"
echo "model        : $model"
echo "gpu(s)       : $gpu_ids  ($gpu_type)"
echo "gpu mem      : $gpu_mem"
echo "interconnect : $interconnect"
echo "tp           : $tp"
echo "runs         : $runs"
echo "output       : $outdir"
echo "========================================"
echo ""


if ["$dry_run" != "true"]; then
    
fi


# run sweep
CUDA_VISIBLE_DEVICES=$gpu_ids vllm bench sweep serve \
    --serve-cmd "vllm serve ${model} \
    --tensor-parallel-size ${TP} \
    --gpu-memory-utilization ${gpu_mem} ${serve_extra}" \
    --bench-cmd "vllm bench serve \
    --model ${model} ${bench_extra}" \
    --num-runs $runs \
    --bench-params "$bench_params_file" \
    --serve-params "$serve_params_file" \
    --output-dir "$outdir"
    

# log experiment
log="${BASE_outdir}/experiment_log.csv"
daye=$(date +%d-%m-%Y)
if [ ! -f "$log" ]; then
    echo "date,experiment,experiment_type,gpu_type,model,tp,outdir,config" > "$log"
fi
echo "${date},${experiment},${experiment_type},${gpu_type},${model},${TP},${outdir},${CONFIG}" >> "$LOG"

echo "========================================"
echo "benchmark suite completed!"
echo "results saved to: $outdir"
echo "experiment log updated: $LOG"