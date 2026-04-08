#%% ======================================================================
# imports & setup
# ========================================================================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from plot_utils import *

set_figure_style()

#%% ======================================================================
# data loading
# ========================================================================
tp2_path = "../results/multi-gpu/concurrency-sweep/a100/aggregated_concurrency_sweep.csv"
tp1_path = "../results/single-gpu/concurrency-sweep/fine-grained/averaged_concurrency_sweep_fine_grained_results.csv"

tp2_data= pd.read_csv(tp2_path)
tp1_data= pd.read_csv(tp1_path)

# ========================================================================
# preprocessing
# ========================================================================
def average_results(df):
    """ average sweep results across runs of same configs """
    df = df.copy()

    # columns that identify a configuration
    group_cols = [
        'max_concurrency',
        'model_id', 
        'tokenizer_id', 
        'backend', 
        'endpoint_type', 
        'label',
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    # drop redundant columns
    drop_cols = ['Unnamed: 0', 'date', 'run_number', 'num_prompts',]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # average remain columns by config
    avg_cols = [c for c in df.columns if c not in group_cols]
    averaged = df.groupby(group_cols, dropna=False)[avg_cols]\
        .mean().reset_index()
 
    return averaged

def filter_df(df, col, val):
    return df[df[col] == val].copy()

# average raw data across runs
averaged = average_results(tp2_data)

# filter csv to group data for plotting
tp2_df = filter_df(averaged, 'label', 'mainsweep')
tp2_nwfb_df = filter_df(averaged, 'label', 'networkfallback')
tp1_df = filter_df(tp1_data, 'gpu_type', 'a100')

# ========================================================================
# kv cache saturation points
# ========================================================================

# constants for llama 3.1 8B
_KV_NUM_LAYERS   = 32
_KV_NUM_KV_HEADS = 8
_KV_HEAD_DIM     = 128
_KV_DTYPE_BYTES  = 2    # bf16

def kv_per_request_gb(isl=512, osl=512):
    """ kv cache memory (GB) of one request at mid-decode """
    avg_tokens = isl + osl / 2
    return (2 * _KV_NUM_LAYERS * avg_tokens * _KV_NUM_KV_HEADS
            * _KV_HEAD_DIM * _KV_DTYPE_BYTES) / 1e9

def kv_allocated_gb(
        total_vram, gpu_mem_util, weight_gb=16.0, activation_frac=0.1):
    """ estimated allocated available kv cache size (GB) """
    return total_vram * gpu_mem_util - weight_gb - total_vram * activation_frac

def kv_saturation_point(
        total_vram, gpu_mem_util, isl=512, osl=512, weight_gb=16.0, activation_frac=0.1):
    """ estimated analytical concurrency where kv cache is saturated """
    reserved_kv_gb = kv_allocated_gb(
        total_vram, gpu_mem_util, weight_gb, activation_frac)
    return reserved_kv_gb / kv_per_request_gb(isl, osl)

# compute saturation lines
tp1_a100_saturation = kv_saturation_point(40, 0.85)
tp1_v100_saturation = kv_saturation_point(32, 0.85)

tp2_a100_saturation = kv_saturation_point(80, 0.85)
tp2_v100_saturation = kv_saturation_point(64, 0.85)

tp4_a100_saturation = kv_saturation_point(160, 0.85)

print(tp1_a100_saturation)
print(tp1_v100_saturation)
print(tp2_a100_saturation)
print(tp4_a100_saturation)

# ========================================================================
# plotting
# ========================================================================
def plot_metric_vs_concurrency(
    dfs,
    metric,
    ylabel,
    title,
    labels,
    colors,
    paper=False,
):
    fig, ax = plt.subplots()

    for df, label, color in zip(dfs, labels, colors):
        ax.plot(df['max_concurrency'], df[metric], marker='o', 
                label=label, color=color)
   
    ax.set_title(title, pad=20)
    ax.set_xlabel('Concurrency (max in-flight requests)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    return fig

#%% ======================================================================
# TP comparison
# ========================================================================
plot_metric_vs_concurrency(
    [tp2_df, tp1_df],
    metric="mean_ttft_ms",
    ylabel="TTFT (ms)",
    title="TTFT vs Concurrency",
    labels=["TP=2", "TP=1"],
    colors=["#4C9BE8", "#E85C4C"],
)
plot_metric_vs_concurrency(
    [tp2_df, tp1_df],
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL vs Concurrency",
    labels=["TP=2", "TP=1"],
    colors=["#4C9BE8", "#E85C4C"],
)
plot_metric_vs_concurrency(
    [tp2_df, tp1_df],
    metric="output_throughput",
    ylabel="Output Throughput",
    title="Output Throughput",
    labels=["TP=2", "TP=1"],
    colors=["#4C9BE8", "#E85C4C"],
)

#%% ======================================================================
# nvlink vs network fallback
# ========================================================================
plot_metric_vs_concurrency(
    [tp2_df, tp2_nwfb_df],
    metric="mean_ttft_ms",
    ylabel="TTFT (ms)",
    title="TTFT vs Concurrency",
    labels=["NVLink", "Network Socket"],
    colors=["#4C9BE8", "#E85C4C"],
)
plot_metric_vs_concurrency(
    [tp2_df, tp2_nwfb_df],
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL vs Concurrency",
    labels=["NVLink", "Network Socket"],
    colors=["#4C9BE8", "#E85C4C"],
)
plot_metric_vs_concurrency(
    [tp2_df, tp2_nwfb_df],
    metric="output_throughput",
    ylabel="Output Throughput",
    title="Output Throughput",
    labels=["NVLink", "Network Socket"],
    colors=["#4C9BE8", "#E85C4C"],
)
