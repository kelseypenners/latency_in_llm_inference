#!/bin/bash
# measure_interconnect.sh

set -eu

# paths to nccl-tests and results
NCCL_TESTS="../../nccl-tests/build/all_reduce_perf"
RESULTS_DIR="../results/interconnect_characterization"
P2P_TESTS="../../cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest"

#-----------------------------------------------
# helpers
#-----------------------------------------------
usage() {
    cat <<EOF

usage: ./$(basename "$0") [required] [options]

required:
  --gpu-ids  <ids>      comma-separated CUDA device IDs
  --tp       <n>        tensor parallel degree
  --server   <name>     server name for labeling output

options:
  --iters      <n>      benchmark iterations per message size  (default: 20)
  --warmup     <n>      warmup iterations  (default: 5)
  --help                show this message

EOF
    exit 0
}

# exit with error
die() { echo "error: $*" >&2; exit 1; }

#-----------------------------------------------
# parse and validate args
#-----------------------------------------------
gpu_ids=""
tp=""
server=""
iters=20
warmup=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-ids)   gpu_ids="$2";   shift 2;;
        --tp)        tp="$2";        shift 2;;
        --server)    server="$2";    shift 2;;
        --iters)     iters="$2";     shift 2;;
        --warmup)    warmup="$2";    shift 2;;
        --help) usage;;
        *) die "unknown argument: $1. use --help for usage.";;
    esac
done

[[ -z "$gpu_ids" ]] && die "missing required: --gpu-ids"
[[ -z "$tp"      ]] && die "missing required: --tp"
[[ -z "$server"  ]] && die "missing required: --server"
[[ ! -f "$NCCL_TESTS" ]] && die "nccl-tests binary not found: $NCCL_TESTS"
[[ ! -f "$P2P_TESTS" ]] && die "p2pBandwidthLatencyTest binary not found: $P2P_TESTS"


# min / max message calculation
DTYPE_BYTES=2
# min message size (1B hidden dim = 2048)
MIN_BYTES=$(( 1 * 1 * 2048 * DTYPE_BYTES ))
# max message size (13B hidden dim = 5120)
MAX_BYTES=$(( 2048 * 5120 * DTYPE_BYTES))

# make outdir and file labels
mkdir -p "$RESULTS_DIR"
timestamp=$(date +%Y%m%d_%H%M)
run_label="${server}_tp${tp}_${timestamp}"
p2p_label="${server}_${timestamp}"

#-----------------------------------------------
# run nccl & p2p tests!
#-----------------------------------------------
run_nccl_test() {
    local outfile="$1"
    shift 1
    local env_vars=("$@")

    env CUDA_VISIBLE_DEVICES="$gpu_ids" \
        "${env_vars[@]}" \
        "$NCCL_TESTS" \
            -b "$MIN_BYTES" -e "$MAX_BYTES" \
            -f 2 -g "$tp" \
            -n "$iters" -w "$warmup" \
            -c 1 \
        2>&1 | tee "$outfile"
    
    echo "saved: $outfile"
    echo ""

}

run_p2p_test() {
    local outfile="$1"

    "$P2P_TESTS" 2>&1 | tee "$outfile"

    echo "saved $outfile"
    echo ""
}

echo ""
echo "========================================"
echo "server=$server | gpu_ids=$gpu_ids | tp=$tp"
echo "range: $MIN_BYTES - $MAX_BYTES bytes"
echo "========================================"

run_p2p_test \
    "${RESULTS_DIR}/${p2p_label}_p2p_matrix.txt"

# default config: NCCL chooses best path
run_nccl_test \
    "${RESULTS_DIR}/${run_label}_default.txt"

# p2p disabled: falls back to SHM through CPU memory via PCIe
run_nccl_test \
    "${RESULTS_DIR}/${run_label}_shm.txt" \
    NCCL_P2P_DISABLE=1

# p2p + shm disabled: falls back to TCP socket over network socket
run_nccl_test \
    "${RESULTS_DIR}/${run_label}_socket.txt" \
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1

#-----------------------------------------------
# print summary
#-----------------------------------------------
echo ""
echo "done! results in $RESULTS_DIR:"
echo "    ${run_label}_default.txt"
echo "    ${run_label}_shm.txt"
echo "    ${run_label}_socket.txt"
echo "    ${p2p_label}_p2p_matrix.txt"
echo "================================================================================"