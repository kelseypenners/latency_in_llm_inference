#!/bin/bash
# run_experiment.sh
# usage: ./run_experiment.sh --experiment <name> --hardware <name> --model <name> --gpu-ids <ids> [options]

set -eu

CONFIGS_DIR="$(dirname "$0")/configs"
RESULTS_DIR="../results"

# default config
experiment=""
hardware=""
model=""
interconnect="default"
gpu_ids=""
tp=1
runs="1"
dry_run=false
resume=""
nccl_debug=""

# parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment) experiment="$2"; shift 2;;
        --hardware) hardware="$2"; shift 2 ;;
        --model) model="$2"; shift 2;;
        --interconnect) interconnect="$2"; shift 2;;
        --gpu-ids) gpu_ids="$2"; shift 2;;
        --tp) tp="$2"; shift 2;;
        --runs) runs="$2"; shift 2;;
        --dry-run) dry_run=true; shift;;
        --resume) resume="$2"; shift 2;;
        --nccl-debug) nccl_debug="$2"; shift 2;;
        *)
            echo "unknown argument: $1"
            echo "usage: ./run_experiment.sh --experiment <name> --hardware <name> --model <name> --gpu-ids <ids>" 
            echo "                          [--interconnect <name>] [--tp <n>] [--runs <n>] [--dry-run] [--resume <timestamp>] [--nccl-debug <level>]"
            exit 1
            ;;
    esac
done

# check for required args
missing=()
[ -z "$experiment" ] && missing+=("--experiment")
[ -z "$hardware" ] && missing+=("--hardware")
[ -z "$model" ] && missing+=("--model")
[ -z "$gpu_ids" ]  && missing+=("--gpu-ids")

if [ ${#missing[@]} -gt 0 ]; then
    echo "error: missing required arguments: ${missing[*]}"
    exit 1
fi

# read config files
experiment_config="${CONFIGS_DIR}/experiments/${experiment}.json"
hardware_config="${CONFIGS_DIR}/hardware/${hardware}.json"
model_config="${CONFIGS_DIR}/models/${model}.json"
interconnect_config="${CONFIGS_DIR}/interconnect/${interconnect}.json"

for f in "$experiment_config" "$hardware_config" "$model_config" "$interconnect_config"; do
    if [ ! -f "$f" ]; then
        echo "error: config file not found: $f"
        exit 1
    fi
done

# merge configs
merged=$(jq -s '.[0] * .[1] * .[2] * .[3]' \
    "$experiment_config" "$hardware_config" "$model_config" "$interconnect_config")

# read merged config values
experiment=$(echo "$merged" | jq -r '.experiment')
gpu_type=$(echo "$merged" | jq -r '.gpu_type')
gpu_mem=$(echo "$merged" | jq -r '.gpu_mem // 0.85')
model_path=$(echo "$merged" | jq -r '.model')
model_name=$(echo "$merged" | jq -r '.model_name')
serve_extra=$(echo "$merged" | jq -r '.serve_cmd_extra // ""')
bench_extra=$(echo "$merged" | jq -r '.bench_cmd_extra // ""')
disable_p2p=$(echo "$merged" | jq -r '.disable_p2p // false')
network_fallback=$(echo "$merged" | jq -r '.network_fallback // false')
config_nccl_debug=$(echo "$merged" | jq -r '.nccl_debug // ""')

# create out dir
outdir="${RESULTS_DIR}/${experiment}/${model_name}/${gpu_type}/tp${tp}/${interconnect}"
mkdir -p "$outdir"

# create run metadata dir
timestamp=$(date +%Y%m%d_%H%M%S)
meta_dir="${outdir}/run_${timestamp}"
mkdir -p "$meta_dir"

label="${experiment}_tp${tp}_${interconnect}"

# write params files
bench_params_flag=""
serve_params_flag=""
if echo "$merged" | jq -e '.bench_params' > /dev/null 2>&1; then
    bench_params_file=$(mktemp /tmp/bench_params_XXXX.json)
    echo "$merged" | jq '.bench_params' > "$bench_params_file"
    bench_params_flag="--bench-params $bench_params_file"
fi
if echo "$merged" | jq -e '.serve_params' > /dev/null 2>&1; then
    serve_params_file=$(mktemp /tmp/serve_params_XXXX.json)
    echo "$merged" | jq '.serve_params' > "$serve_params_file"
    serve_params_flag="--serve-params $serve_params_file"
fi

# NCCL env vars for interconnect control
nccl_env=()
nccl_comm="default (best available)"

# cli overrides config
if [ -z "$nccl_debug" ]; then
    nccl_debug="$config_nccl_debug"
fi

if [ "$disable_p2p" = true ]; then
    # disable using NCCL_P2P_DISABLE
    nccl_env=(NCCL_P2P_DISABLE=1)
    nccl_comm="P2P disabled (SHM transport)" 
fi
if [ "$network_fallback" = true ]; then
    # disable using NCCL_P2P_DISABLE and NCCL_SHM_DISABLE
    nccl_env=(NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1)
    nccl_comm="network fallback (P2P + SHM disabled, socket transport)"
fi
if [ -n "$nccl_debug" ]; then
    # enable NCCL_DEBUG
    nccl_env+=(NCCL_DEBUG=$nccl_debug)
    nccl_env+=(NCCL_DEBUG_FILE="${meta_dir}/nccl_debug_%h_%p.log")
fi

# label flag
label_flag=""
if [ "$label" != "" ]; then
    label_flag="--label $label"
fi

# dry run flag
dry_run_flag=""
if [ "$dry_run" = true ]; then
    dry_run_flag="--dry-run"
fi

# resume flag
resume_flag=""
if [ "$resume" != "" ]; then
    resume_flag="--resume $resume"
fi

# save merged config
echo "$merged" | jq \
    --arg gpu_ids "$gpu_ids" \
    --argjson tp "$tp" \
    --argjson runs "$runs" \
    --arg interconnect "$interconnect" \
    '. + {gpu_ids: $gpu_ids, tp: $tp, runs: $runs, interconnect: $interconnect}' \
    > "${meta_dir}/config_merged.json"


# GPU power logging
timestamp=$(date +%d-%m-%Y_%H-%M-%S)
power_log="${meta_dir}/gpu_power.csv"

# log every 5 seconds
nvidia-smi \
  --query-gpu=timestamp,index,power.draw \
  --format=csv,noheader,nounits \
  -l 5 > "$power_log" &
POWER_LOG_PID=$!

# cleanup temp files
cleanup() {
    rm -f "${bench_params_file:-}" "${serve_params_file:-}"
    if ps -p $POWER_LOG_PID > /dev/null 2>&1; then
        kill $POWER_LOG_PID
        wait $POWER_LOG_PID 2>/dev/null
        echo "power logging stopped"
    fi
}
trap cleanup EXIT

# print summary
echo "================================================================================"
echo "experiment    : $experiment  (label = $label)"
echo "model         : $model_name ($model_path)"
echo "gpu id(s)     : $gpu_ids  ($gpu_type)"
echo "gpu mem       : $gpu_mem"
echo "communication : $nccl_comm"
echo "tp            : $tp"
echo "runs          : $runs"
echo "output        : $outdir"
if [ -n "$nccl_debug" ]; then
    echo "nccl debug    : $nccl_debug" 
fi
if [ "$dry_run" = "true" ]; then
    echo "mode          : DRY RUN"
fi
if [ "$resume" != "" ]; then
    echo "resuming run  : $resume"
fi
echo "================================================================================"
echo ""

# run sweep
env CUDA_VISIBLE_DEVICES=$gpu_ids "${nccl_env[@]}" \
    vllm bench sweep serve \
    --serve-cmd "vllm serve ${model_path} \
        --tensor-parallel-size ${tp} \
        --gpu-memory-utilization ${gpu_mem} ${serve_extra}" \
    --bench-cmd "vllm bench serve \
        --model ${model_path} ${label_flag} ${bench_extra}" \
    --num-runs "$runs" \
    $bench_params_flag $serve_params_flag \
    --output-dir "$meta_dir" \
    $dry_run_flag $resume_flag

# log experiment
log="${RESULTS_DIR}/experiment_log.csv"
date_str=$(date +%d-%m-%Y)
if [ ! -f "$log" ]; then
    echo "date,experiment,model_name,gpu_type,tp,interconnect,gpu_ids,outdir" > "$log"
fi
echo "${date_str},${experiment},${model_name},${gpu_type},${tp},${interconnect},${gpu_ids},${outdir}" >> "$log"

echo "========================================"
echo "benchmark suite completed!"
echo "results saved to: $outdir"
echo "experiment log updated: $log"