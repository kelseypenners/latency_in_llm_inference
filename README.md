# LLM Inference Benchmarking

Thesis repo for benchmarking LLM inference performance across single and multi-GPU configurations. The main question is how inter-GPU communication latency (NVLink vs PCIe vs network socket) affects throughput, TTFT, ITL, and other performance metrics when an LLM is distributed across several devices. 

**Framework:** vLLM 0.11.2 \
**Models:** Llama-3.2-1B, Llama-3.1-8B, Llama-2-13B \
**Hardware:** shy-fec (8 × A100, NVLink), funnotch (4 × V100, PCIe)

---

## Structure

```
benchmarking/       run_experiment.sh + JSON configs
results/            raw benchmark output (not committed)
analysis/           parsing and derived metrics
visualizing/        plotting scripts
utils/              misc helpers
archive/            old scripts and archived data
``` 

## Requirements

- vLLM 0.11.2
- Python: *pandas, numpy, matplotlib, seaborn, scikit-learn*
- *jq* for config merging in shell script
- r`equirements.txt` holds actual versions used

---

## Running an experiment

```bash
cd benchmarking
./run_experiment.sh \ 
    --experiment <name> \
    --hardware <name> \
    --model <name> \ 
    --gpu-ids <ids> \
    [--tp <n>] [--interconnect <name>] [--runs <n>] \
    [--dry-run] [--nccl-debug <mode>] [--help]
```

The script merges four JSON configs (experiment + hardware + model + interconnect), sets any NCCL environment variables and launces vLLM benchmarking using given parameters. 

**Some examples:**

Single GPU:
```bash
./run_experiment.sh --experiment concurrency_sweep --hardware a100 -model llama_8b --gpu-ids 0
```

TP=2 with default GPU communication:
```bash
./run_experiment.sh --experiment concurrency_sweep --hardware a100 -model llama_8b --gpu-ids 0,2 --tp 2
```

TP=2 forcing network socket transport (P2P + SHM disabled):
```bash
./run_experiment.sh --experiment concurrency_sweep --hardware a100 -model llama_8b --gpu-ids 0,2 --tp 2 --interconnect network_socket
```

Add `--nccl-debug INFO` to any multi-GPU run to get NCCL debug logs, which is useful for verifying which transport is actually being used between GPUs.

Add `--dry-run` to simulate the command without actually running anything yet. 

Add `--resume {vllm_timestamp}` to resume a vLLM sweep from the last detected checkpoint.

---

## Interconnect configs

There are three transport configurations for multi-GPU runs:

| `--interconnect`  | NCCL env vars                                | Transport      |
|-------------------|----------------------------------------------|----------------|
| default          | none                                         | NVLink / PCIe  |
| p2p_disabled     | `NCCL_P2P_DISABLE=1`                         | SHM            |
| network_socket   | `NCCL_P2P_DISABLE=1` + `NCCL_SHM_DISABLE=1` | TCP socket     |

NCCL selects the fastest available transport between GPUs by default: NVLink when present (on shy-fec), PCIe otherwise (on funnotch). The three interconnect configs control which transport NCCL uses by disabling faster paths. Setting `NCCL_P2P_DISABLE=1` disables direct GPU-to-GPU transfers and forces traffic through shared CPU memory (SHM) via the PCIe bus. Adding `NCCL_SHM_DISABLE=1` on top removes the shared memory path, forcing a fall back to TCP socket over the network stack.

The transport actually used can be verified by inspecting NCCL debug logs, looking for `P2P/CUMEM`, `SHM/direct/direct`, `NET/Socket/0`, etc..

---

## Results structure

```
results/{experiment}/{model}/{gpu_type}/tp{n}/{interconnect}/run_{timestamp}/
```

Each run directory has a `config_merged.json` (what was actually run), `gpu_power.csv`, any NCCL debug logs, and the vLLM output with per run `summary.json` files.

Once results are in, the pipeline used is:

1. **Parse results** - `analysis/parse_results.py` which collects all summary files for an experiments, pulls data from the results path, and outputs an aggregated and an averaged CSV. 

2. **Derive additional metrics** - `analysis/data_analysis.py` merges multi-GPU configs against the TP=1 baseline to compute other metrics like throughput efficiency and latency metric slowdown. 

3. **Plots** - `visualizing/single_gpu_plotting.py` and `multi_gpu_plotting.py` plot results by loading averaged CSVs. 

---
