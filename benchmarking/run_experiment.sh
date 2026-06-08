#!/bin/bash
# run_experiment.sh

set -eu

CONFIGS_DIR="$(dirname "$0")/configs"
RESULTS_DIR="../results"

#-----------------------------------------------
# helpers
#-----------------------------------------------

usage() {
    cat <<EOF

usage: ./$(basename "$0") [required flags] [options]
 
required:
  --experiment <name>      experiment config name (configs/experiments/<name>.json)
  --hardware   <name>      hardware config name   (configs/hardware/<name>.json)
  --model      <name>      model config name      (configs/models/<name>.json)
  --gpu-ids    <ids>       comma-separated CUDA device IDs
 
options:
  --interconnect <name>    interconnect config (default up to NCCL)
  --tp <n>                 tensor parallel degree (default: 1)
  --runs <n>               number of runs per config (default: 1)
  --nccl-debug <level>     enable NCCL debug logging (e.g. INFO, WARN)
  --resume <timestamp>     resume a previous run by its vLLM output timestamp
  --dry-run                print commands without running
  --help                   show this message

EOF
    exit 0
}

# exit with error
die() { echo "error: $*" >&2; exit 1; }

#-----------------------------------------------
# parse and validate args
#-----------------------------------------------

# default config
experiment=""
hardware=""
model=""
interconnect="default"
gpu_ids=""
tp=1
runs=1
nccl_debug=""
resume=""
dry_run=false

# parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment) experiment="$2"; shift 2;;
        --hardware) hardware="$2"; shift 2 ;;
        --model) model="$2"; shift 2;;
        --gpu-ids) gpu_ids="$2"; shift 2;;
        --interconnect) interconnect="$2"; shift 2;;
        --tp) tp="$2"; shift 2;;
        --runs) runs="$2"; shift 2;;
        --nccl-debug) nccl_debug="$2"; shift 2;;
        --resume) resume="$2"; shift 2;;
        --dry-run) dry_run=true; shift;;
        --help) usage ;;
        *) die "unknown argument: $1. use --help for usage!" ;;
    esac
done

# check for required args
missing=()
[[ -z "$experiment" ]] && missing+=("--experiment")
[[ -z "$hardware" ]] && missing+=("--hardware")
[[ -z "$model" ]] && missing+=("--model")
[[ -z "$gpu_ids" ]] && missing+=("--gpu-ids")
[[ ${#missing[@]} -gt 0 ]] && die "missing required arguments: ${missing[*]}"

# [[ "$tp" -eq 1 ]] && interconnect="default"

# read config files
experiment_config="${CONFIGS_DIR}/experiments/${experiment}.json"
hardware_config="${CONFIGS_DIR}/hardware/${hardware}.json"
model_config="${CONFIGS_DIR}/models/${model}.json"
interconnect_config="${CONFIGS_DIR}/interconnect/${interconnect}.json"

for f in "$experiment_config" "$hardware_config" "$model_config" "$interconnect_config"; do
    [[ ! -f "$f" ]] && die "config file not found: $f"
done

#-----------------------------------------------
# merge configs and get values
#-----------------------------------------------

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

# CLI --nccl_debug overrides config value
[[ -z "$nccl_debug" ]] && nccl_debug="$config_nccl_debug"

#-----------------------------------------------
# write temp params file
#-----------------------------------------------

bench_params_file=""
serve_params_file=""

# clean up temp files on exit
cleanup() {
    rm -f "$bench_params_file" "$serve_params_file"
}
trap cleanup EXIT

if echo "$merged" | jq -e '.bench_params' > /dev/null 2>&1; then
    bench_params_file=$(mktemp /tmp/bench_params_XXXX.json)
    echo "$merged" | jq '.bench_params' > "$bench_params_file"
fi
if echo "$merged" | jq -e '.serve_params' > /dev/null 2>&1; then
    serve_params_file=$(mktemp /tmp/serve_params_XXXX.json)
    echo "$merged" | jq '.serve_params' > "$serve_params_file"
fi

#-----------------------------------------------
# NCCL environment variables
#-----------------------------------------------

nccl_env=()
nccl_comm="default (best available)"
env_vars_active=false
if [[ "$disable_p2p" == true ]]; then
    # disable using NCCL_P2P_DISABLE
    nccl_env+=(NCCL_P2P_DISABLE=1)
    nccl_comm="SHM (P2P disabled)" 
    env_vars_active=true
fi
if [[ "$network_fallback" == true ]]; then
    # disable using NCCL_P2P_DISABLE and NCCL_SHM_DISABLE
    nccl_env=(NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1)
    nccl_comm="socket (P2P + SHM disabled)"
    env_vars_active=true
fi
if [[ "$tp" -eq 1 ]]; then
    # clarify no gpu to gpu comm in tp1, but allow for sanity checks
    if [[ "$env_vars_active" == true ]]; then
        nccl_comm="env vars set, but no gpu-to-gpu communication in this config"
    else
        nccl_comm=""
    fi
fi

#-----------------------------------------------
# output directory
# results/{experiment}/{model_name}/{gpu_type}/tp{n}/{interconnect}/run_{TS}
# vLLM will create TS'd subdir here
#-----------------------------------------------

outdir="${RESULTS_DIR}/${experiment}/${model_name}/${gpu_type}/tp${tp}/${interconnect}"
timestamp=$(date +%Y%m%d_%H%M%S)
run_dir="${outdir}/run_${timestamp}"
label="${experiment}_tp${tp}_${interconnect}"

#-----------------------------------------------
# print summary
#-----------------------------------------------

echo "================================================================================"
echo "experiment    : $experiment"
echo "model         : $model_name ($model_path)"
echo "gpu id(s)     : $gpu_ids  ($gpu_type)"
echo "gpu mem       : $gpu_mem"
echo "tp            : $tp"
[[ -n "$nccl_comm" ]] && echo "interconnect  : $nccl_comm"
echo "runs          : $runs"
echo "output dir    : $outdir"
[[ -n "$nccl_debug" ]] && echo "nccl debug    : $nccl_debug" 
[[ -n "$resume" ]] && echo "resuming run  : $resume"
[[ "$dry_run" == true ]] && echo "mode          : DRY RUN"
echo "================================================================================"
echo ""

#-----------------------------------------------
# build vllm command fragments
#-----------------------------------------------

serve_cmd="vllm serve ${model_path} \
    --tensor-parallel-size ${tp} \
    --gpu-memory-utilization ${gpu_mem} ${serve_extra}"

bench_cmd="vllm bench serve \
    --model ${model_path} \
    --label ${label} ${bench_extra}"

sweep_cmd=(
    vllm bench sweep serve
    --serve-cmd "$serve_cmd"
    --bench-cmd "$bench_cmd"
    --num-runs "$runs"
)
[[ -n "$bench_params_file" ]] && sweep_cmd+=(--bench-params "$bench_params_file")
[[ -n "$serve_params_file" ]] && sweep_cmd+=(--serve-params "$serve_params_file")

#-----------------------------------------------
# dry run
#-----------------------------------------------

if [[ "$dry_run" == true ]]; then
    env CUDA_VISIBLE_DEVICES="$gpu_ids" "${nccl_env[@]}" \
        "${sweep_cmd[@]}" --dry-run
    exit 0
fi

#-----------------------------------------------
# make outdir
# create rundir or resume from existing
# save merged config
#-----------------------------------------------

mkdir -p "$outdir"

# create or locate run dir
if [[ -n "$resume" ]]; then
    run_dir=""
    # search for vLLM resume dir
    for d in "${outdir}"/run_*; do
        if [[ -d "$d/$resume" ]]; then
            run_dir="$d"
            break
        fi
    done

    [[ -z "$run_dir" ]] && die "could not find resume run '$resume'"
    echo "found resume directory: $run_dir"
    sweep_cmd+=(--resume "$resume")
else
    mkdir -p "$run_dir"
fi

# save merged config in results dir
echo "$merged" | jq \
    --arg gpu_ids "$gpu_ids" \
    --argjson tp "$tp" \
    --argjson runs "$runs" \
    --arg interconnect "$interconnect" \
    '. + {gpu_ids: $gpu_ids, tp: $tp, runs: $runs, interconnect: $interconnect}' \
    > "${run_dir}/config_merged.json"

#-----------------------------------------------
# GPU power & NCCL logging
#-----------------------------------------------

# NCCL logging
if [[ -n "$nccl_debug" ]]; then
    nccl_env+=(NCCL_DEBUG=$nccl_debug)
    nccl_env+=(NCCL_DEBUG_FILE="${run_dir}/nccl_debug_%h_%p.log")
fi

power_log="${run_dir}/gpu_power.csv"

if [ ! -s "$power_log" ]; then
    echo "timestamp,index,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total" > "$power_log"
fi

# measure power at 20 hz
nvidia-smi \
  --query-gpu=timestamp,index,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits \
  -lms 50 >> "$power_log" &
POWER_LOG_PID=$!

stop_power_log() {
    cleanup # also remove temp files
    if [[ -n "${POWER_LOG_PID:-}" ]] && ps -p "$POWER_LOG_PID" > /dev/null 2>&1; then
        kill "$POWER_LOG_PID"
        wait "$POWER_LOG_PID" 2>/dev/null
        echo "power logging stopped"
    fi
}
trap stop_power_log EXIT

#-----------------------------------------------
# run sweep
#-----------------------------------------------

env CUDA_VISIBLE_DEVICES="$gpu_ids" "${nccl_env[@]}" \
    "${sweep_cmd[@]}" --output-dir "$run_dir" 

#-----------------------------------------------
# log experiment
#-----------------------------------------------

log="${RESULTS_DIR}/experiment_log.csv"
date_str=$(date +%d-%m-%Y)
if [[ ! -f "$log" ]]; then
    echo "date,experiment,model_name,gpu_type,tp,interconnect,gpu_ids,outdir" > "$log"
fi
echo "${date_str},${experiment},${model_name},${gpu_type},${tp},${interconnect},${gpu_ids},${run_dir}" >> "$log"
echo "========================================"
echo "benchmark suite completed!"
echo "results saved to: $outdir"
echo "experiment log updated: $log"