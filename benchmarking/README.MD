# Experiment Configs

`run_experiment.sh` builds its full configuration by merging four JSON files with *jq*:

```
configs/
  experiments/    what to benchmark (bench params, sweep ranges)
  hardware/       GPU-specific settings (gpu_type, gpu_memory_utilization)
  models/         model path and name
  interconnect/   NCCL transport overrides (p2p disable, shm disable)
```

The merged result is saved as `config_merged.json` in the run output directory for reproducibility.

---

## Config Types

### `experiments/`

Defines the benchmark sweep, what `vllm bench sweep serve` should vary. \
Experiment types include a concurrency sweep across various max concurrencies, a sequence length sweep measuring performance with various input and output sequence lengths, and a dataset comparison that compares various prompt types (random, repeating, coherent) 

```json
{
  "experiment": "concurrency-sweep",
  "bench_params": [
    {"max-concurrency": 4,   "num-prompts": 200},
    {"max-concurrency": 8,   "num-prompts": 200},
    {"max-concurrency": 16,  "num-prompts": 200},
    {"max-concurrency": 32,  "num-prompts": 500}
  ],
  "bench_cmd_extra": "--dataset-name random --random-input-len 512 --random-output-len 512"
}
```
Fields:
- `experiment` — used as the top-level results directory name
- `bench_params` — list of `vllm bench serve` parameter combinations to sweep
- `bench_cmd_extra` — extra flags added to every `vllm bench serve` call
- `serve_params` — list of `vllm serve` parameter combinations to sweep
- `serve_cmd_extra` — extra flags appended to every `vllm serve` call

### `hardware/`

GPU-specific serve settings.

```json
{
  "gpu_type": "a100",
  "gpu_mem": 0.85
}
```

Fields:
- `gpu_type` — used to organize data in the results directory
- `gpu_mem` — passed as `--gpu-memory-utilization` to `vllm serve`

### `models/`

Model path and name.

```json
{
  "model": "../Llama-3.1-8B",
  "model_name": "llama-3.1-8b"
}
```

Fields:
- `model` — path passed to `vllm serve`
- `model_name` — used to organize data in the results directory

### `interconnect/`

Controls NCCL transport by setting environment variables before launching vLLM.

```json
{
  "disable_p2p": true,
  "network_fallback": false
}
```

Key fields:
- `disable_p2p: true` — sets `NCCL_P2P_DISABLE=1`, forces SHM transport
- `network_fallback: true` — sets `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1`, forces TCP socket transport
- `nccl_debug` — (optional) sets `NCCL_DEBUG` level, e.g. `"INFO"`, and writes logs to the run output directory

The `default.json` sets both flags to false (no overrides, so NCCL picks the best available transport).

---

## Adding a New Config

To add a new experiment (e.g. a sequence length sweep at higher concurrency, test other datasets,  etc.):

1. Create `configs/experiments/new_experiment.json` with desired sweep parameters
2. Run with an existing hardware, model, and interconnect config (or build your own!):
   ```bash
   ./run_experiment.sh \
     --experiment new_experiment \
     --hardware a100 \
     --model llama_8b \
     --gpu-ids 0
   ```

Results land in `results/new_experiment/llama-3.1-8b/a100/tp1/default/run_{timestamp}/`.

To add a new hardware target (e.g. a new server), create `configs/hardware/newserver.json` with `gpu_type` and `gpu_mem`. No other files need to change.

To add a new model, create `configs/models/new_model.json` with the new model path under `model` and `model_name`. Again, no other files need to change.