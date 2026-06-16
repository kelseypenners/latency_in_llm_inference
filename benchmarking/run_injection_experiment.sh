#!/bin/bash
# run_injection_experiment.sh

set -eu

CONFIGS_DIR="$(dirname "$0")/configs"

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
  --tp <n>                 tensor parallel degree (default: 2)
  --runs <n>               number of runs per config (default: 1)
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
gpu_ids=""
tp=2
runs=1
resume=""
dry_run=false

# parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment) experiment="$2"; shift 2;;
        --hardware) hardware="$2"; shift 2 ;;
        --model) model="$2"; shift 2;;
        --gpu-ids) gpu_ids="$2"; shift 2;;
        --tp) tp="$2"; shift 2;;
        --runs) runs="$2"; shift 2;;
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

# read experiment config file to extract latencies for injection
experiment_config="${CONFIGS_DIR}/experiments/${experiment}.json"
[[ ! -f "$experiment_config" ]] && die "experiment config file not found: $experiment_config"

readarray -t latencies < <(jq -r '.sweep_latencies[]' "$experiment_config")

# run sweep for each latency injection value
for us in "${latencies[@]}"; do
    echo "================================================================================"
    echo "!!!   injecting $us us to allreduce operations for following sweep    !!!"
    
    ./run_experiment.sh \
        --experiment "$experiment" \
        --hardware "$hardware" \
        --model "$model" \
        --gpu-ids "$gpu_ids" \
        --tp "$tp" \
        --inject-latency-us "$us"
done