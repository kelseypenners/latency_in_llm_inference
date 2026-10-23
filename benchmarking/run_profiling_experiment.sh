#!/bin/bash
# run_profiling.sh

set -eu

CONFIGS_DIR="$(dirname "$0")/configs"
RESULTS_DIR="../results/nsys_profiles"
DEFAULT_EXPERIMENT="nsys_profiling"

#-----------------------------------------------
# helpers
#-----------------------------------------------

usage() {
    cat <<EOF

usage: ./$(basename "$0") [required flags] [options]
 
required:
  --hardware   <name>      hardware config name   (configs/hardware/<name>.json)
  --model      <name>      model config name      (configs/models/<name>.json)
  --gpu-ids    <ids>       comma-separated CUDA device IDs
 
options:
  --experiment <name>      experiment config name (default: nsys_profiling)
  --interconnect <name>    interconnect config (default: default)
  --tp <n>                 tensor parallel degree (default: 2)
  --nccl-debug <level>     enable NCCL debug logging (e.g. INFO, WARN)
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

experiment="$DEFAULT_EXPERIMENT"
hardware=""
model=""
gpu_ids=""
interconnect="default"
tp=2
nccl_debug=""
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
        --nccl-debug) nccl_debug="$2"; shift 2;;
        --dry-run) dry_run=true; shift;;
        --help) usage ;;
        *) die "unknown argument: $1. use --help for usage!" ;;
    esac
done

# check for required args
missing=()
[[ -z "$hardware" ]] && missing+=("--hardware")
[[ -z "$model" ]] && missing+=("--model")
[[ -z "$gpu_ids" ]] && missing+=("--gpu-ids")
[[ ${#missing[@]} -gt 0 ]] && die "missing required arguments: ${missing[*]}"

# check dependencies
for cmd in nsys jq vllm curl; do
    command -v "$cmd" >/dev/null || die "$cmd not found on PATH"
done

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

merged=$(jq -s '.[0] * .[1] * .[2] * .[3]' \
    "$experiment_config" "$hardware_config" "$model_config" "$interconnect_config")

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
bench_params=$(echo "$merged" | jq -c '.nsys_bench_params // .bench_params // []')

[[ "$(echo "$bench_params" | jq 'length')" -gt 0 ]] || die "no benchmark points found in $experiment_config"

# CLI --nccl_debug overrides config value
[[ -z "$nccl_debug" ]] && nccl_debug="$config_nccl_debug"

# extract port from serve_extra if present
port=$(grep -oP '(?<=--port )\d+' <<< "$serve_extra" || true)
port=${port:-8000}

#-----------------------------------------------
# NCCL environment variables
#-----------------------------------------------

nccl_env=()
nccl_comm="default (best available)"
env_vars_active=false
if [[ "$disable_p2p" == true ]]; then
    nccl_env+=(NCCL_P2P_DISABLE=1)
    nccl_comm="SHM (P2P disabled)" 
    env_vars_active=true
fi
if [[ "$network_fallback" == true ]]; then
    nccl_env=(NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1)
    nccl_comm="socket (P2P + SHM disabled)"
    env_vars_active=true
fi
if [[ "$tp" -eq 1 ]]; then
    if [[ "$env_vars_active" == true ]]; then
        nccl_comm="env vars set, but no gpu-to-gpu communication in this config"
    else
        nccl_comm=""
    fi
fi

#-----------------------------------------------
# output directory
# results/nsys_profiles/{model_name}/{gpu_type}/tp{n}/{interconnect}/run_{TS}
#-----------------------------------------------

outdir="${RESULTS_DIR}/${model_name}/${gpu_type}/tp${tp}/${interconnect}"
timestamp=$(date +%Y%m%d_%H%M%S)
run_dir="${outdir}/run_${timestamp}"
report_base="${run_dir}/nsys_tp${tp}"
server_log="${run_dir}/server.log"

#-----------------------------------------------
# print summary
#-----------------------------------------------

echo "================================================================================"
echo "profiling experiment: $experiment"
echo "model               : $model_name ($model_path)"
echo "gpu id(s)           : $gpu_ids  ($gpu_type)"
echo "gpu mem             : $gpu_mem"
echo "tp                  : $tp"
[[ -n "$nccl_comm" ]] && echo "interconnect        : $nccl_comm"
echo "output dir          : $outdir"
[[ -n "$nccl_debug" ]] && echo "nccl debug          : $nccl_debug" 
[[ "$dry_run" == true ]] && echo "mode                : DRY RUN"
echo "================================================================================"
echo ""

#-----------------------------------------------
# build command fragments
#-----------------------------------------------

read -r -a serve_extra_args <<< "$serve_extra"
read -r -a bench_extra_args <<< "$bench_extra"

serve_cmd=(
    vllm serve "${model_path}"
    --tensor-parallel-size "${tp}"
    --gpu-memory-utilization "${gpu_mem}"
    "${serve_extra_args[@]}"
)

nsys_cmd=(
    nsys profile
    --output="${report_base}"
    --force-overwrite=true
    --trace-fork-before-exec=true
    --cuda-graph-trace=node
    --capture-range=cudaProfilerApi
    --capture-range-end=repeat
    "${serve_cmd[@]}"
)

#-----------------------------------------------
# dry run
#-----------------------------------------------

if [[ "$dry_run" == true ]]; then
    echo "--- dry run commands ---"
    echo "CUDA_VISIBLE_DEVICES=\"$gpu_ids\" VLLM_TORCH_CUDA_PROFILE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn ${nccl_env[*]} ${nsys_cmd[*]}"
    exit 0
fi

#-----------------------------------------------
# make outdir & save merged config
#-----------------------------------------------

mkdir -p "$run_dir"

echo "$merged" | jq \
    --arg gpu_ids "$gpu_ids" \
    --argjson tp "$tp" \
    --arg interconnect "$interconnect" \
    --argjson profiling true \
    '. + {gpu_ids: $gpu_ids, tp: $tp, interconnect: $interconnect, profiling: $profiling}' \
    > "${run_dir}/config_merged.json"

#-----------------------------------------------
# NCCL logging & cleanup
#-----------------------------------------------

if [[ -n "$nccl_debug" ]]; then
    nccl_env+=(NCCL_DEBUG=$nccl_debug)
    nccl_env+=(NCCL_DEBUG_FILE="${run_dir}/nccl_debug_%h_%p.log")
fi

SERVER_PID=""

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "stopping server; waiting for nsys to write the report..."
        kill -INT "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap stop_server EXIT

#-----------------------------------------------
# run profiled server
#-----------------------------------------------

echo "starting profiled server"
echo "  log: $server_log"
CUDA_VISIBLE_DEVICES="$gpu_ids" \
VLLM_TORCH_CUDA_PROFILE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
"${nccl_env[@]}" \
"${nsys_cmd[@]}" > "$server_log" 2>&1 &
SERVER_PID=$!

# wait for health check
echo "waiting for server to become ready..."
until curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        die "server exited before becoming ready; see $server_log"
    fi
    sleep 5
done

echo "server ready. running profiling benchmarks."

#-----------------------------------------------
# run profiling benchmarks loop
#-----------------------------------------------

i=0
while IFS= read -r point; do
    i=$((i + 1))
    mapfile -t point_flags < <(jq -r 'to_entries[] | "--\(.key)=\(.value|tostring)"' <<< "$point")
    label="${experiment}_tp${tp}_p${i}"
    echo "benchmark $i: $label"
    
    VLLM_TORCH_CUDA_PROFILE=1 vllm bench serve \
        --model "$model_path" \
        --backend openai \
        --base-url "http://localhost:${port}" \
        --label "$label" \
        --result-dir "$run_dir" \
        "${bench_extra_args[@]}" \
        "${point_flags[@]}" \
        --profile
done < <(echo "$bench_params" | jq -c '.[]')

# stop server cleanly to flush/write reports
stop_server
trap - EXIT

# check reports
shopt -s nullglob
reports=("${report_base}"*.nsys-rep)
[[ ${#reports[@]} -gt 0 ]] || die "NSys did not create a report; check $server_log"

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
echo "profiling suite completed!"
echo "results saved to: $run_dir"
echo "nsys reports:"
printf '  %s\n' "${reports[@]}"
echo "experiment log updated: $log"