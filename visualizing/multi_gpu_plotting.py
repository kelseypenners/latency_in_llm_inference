#%% ======================================================================
# imports & setup
# ========================================================================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from plot_utils import *

set_figure_style()

# ========================================================================
# data loading
# ========================================================================
multigpu_path = "../results/multi-gpu/concurrency-sweep/a100/averaged_concurrency_sweep.csv"
singlegpu_path = "../results/single-gpu/concurrency-sweep/fine-grained/averaged_concurrency_sweep_fine_grained_results.csv"

multigpu_data= pd.read_csv(multigpu_path)
singlegpu_data= pd.read_csv(singlegpu_path)

derived_metrics_path = "../results/multi-gpu/concurrency-sweep/a100/derived_metrics.csv"
derived = pd.read_csv(derived_metrics_path)
# ========================================================================
# preprocessing
# ========================================================================
def filter_df(df, col, val):
    return df[df[col] == val].copy()

# filter csv to group data for plotting
tp2_df = filter_df(multigpu_data, 'label', 'mainsweep')
tp2_nwfb_df = filter_df(multigpu_data, 'label', 'networkfallback')
tp2_shm_df = filter_df(multigpu_data, 'label', 'p2pdisabled')
tp4_df = filter_df(multigpu_data, 'label', 'mainsweep_tp4')
tp1_df = filter_df(singlegpu_data, 'gpu_type', 'a100')

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
tp1_a100_saturation = kv_saturation_point(40, 0.85, activation_frac=0.05)
tp1_v100_saturation = kv_saturation_point(32, 0.85)

tp2_a100_saturation = kv_saturation_point(80, 0.85, activation_frac=0.05)
tp2_v100_saturation = kv_saturation_point(64, 0.85)

tp4_a100_saturation = kv_saturation_point(160, 0.85)

# saturation lines for plotting
tp1_a100_line = (tp1_a100_saturation, 'KV saturation:\nTP=1')
tp2_a100_line = (tp2_a100_saturation, 'KV saturation:\nTP=2')
tp4_a100_line = (tp4_a100_saturation, 'KV saturation:\nTP=4')

print(f"A100 saturation point at TP=1: {tp1_a100_saturation}")
print(f"V100 saturation point at TP=1: {tp1_v100_saturation}\n")
print(f"A100 saturation point at TP=2: {tp2_a100_saturation}")
print(f"V100 saturation point at TP=2: {tp2_v100_saturation}\n")
print(f"A100 saturation point at TP=4: {tp4_a100_saturation}")

# ========================================================================
# plotting
# ========================================================================
def plot_metric_vs_concurrency(
    dfs, metric, ylabel, title, labels, colors,
    saturation_lines=None,
    log_scale=None,
    subtitle_info=None,
    paper=False,
):
    fig, ax = plt.subplots()

    for df, label, color in zip(dfs, labels, colors):
        ax.plot(df['max_concurrency'], df[metric], marker='o', 
                label=label, markersize=2, color=color)
   
    if saturation_lines:
        ylim = ax.get_ylim()
        for x_val, sat_label in saturation_lines:
            ax.axvline(x=x_val, color='grey', linewidth=0.9, linestyle=':')
            # ax.text(x_val * 1.04, ylim[1] * 0.97, sat_label,
            #         fontsize=6.5, color='grey', va='top')
 
    ax.set_title(title, pad=20)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_xticks(df['max_concurrency'])
    ax.set_xscale('log', base=2)
    if log_scale:
        ax.set_yscale('log', base=2)
        ticks_s = [0.1, 1, 10, 60]
        ticks_ms = 1000 * np.array(ticks_s)
        ax.set_yticks(ticks_ms)
        ax.set_yticklabels([f"{t}s" for t in ticks_s])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)
    plt.tight_layout()
    return fig

#%% ======================================================================
# TP comparison
# ========================================================================

sub = dict(gpu='shy-fec (A100s)', model='Llama-3.1-8B', config='ISL/OSL = 512/512')

fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp4_df],
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    labels=["TP=1", "TP=2", "TP=4"],
    colors=["#4C9BE8", "#E85C4C", "#5CB85C"],
    log_scale=True,
    saturation_lines=[tp1_a100_line,
                      tp2_a100_line,
                      tp4_a100_line],
    subtitle_info=sub
)
fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp4_df],
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    labels=["TP=1", "TP=2", "TP=4"],
    colors=["#4C9BE8", "#E85C4C", "#5CB85C"],
    saturation_lines=[tp1_a100_line,
                      tp2_a100_line,
                      tp4_a100_line],
    subtitle_info=sub
)
fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp4_df],
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    labels=["TP=1", "TP=2", "TP=4"],
    colors=["#4C9BE8", "#E85C4C", "#5CB85C"],
    subtitle_info=sub
)
fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp4_df],
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    labels=["TP=1", "TP=2", "TP=4"],
    colors=["#4C9BE8", "#E85C4C", "#5CB85C"],
    saturation_lines=[tp1_a100_line,
                      tp2_a100_line,
                      tp4_a100_line],
    subtitle_info=sub
)

#%% ======================================================================
# nvlink vs network fallback vs shm
# ========================================================================

sub = dict(gpu='shy-fec (A100s)', model='Llama-3.1-8B', config='ISL/OSL = 512/512')

fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp2_shm_df, tp2_nwfb_df],
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across Interconnects",
    labels=["Single GPU (TP=1)", "NVLink (TP=2)", "SHM (TP=2)", "Socket (TP=2)"],
    colors=["#4C9BE8", "#E85C4C", "#9B59B6", "#E8A84C"],
    subtitle_info=sub,
)

fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp2_shm_df, tp2_nwfb_df],
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Interconnects",
    labels=["Single GPU (TP=1)", "NVLink (TP=2)", "SHM (TP=2)", "Network Socket (TP=2)"],
    colors=["#4C9BE8", "#E85C4C", "#9B59B6", "#E8A84C"],
    subtitle_info=sub,
)

fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp2_shm_df, tp2_nwfb_df],
    metric="mean_e2el_ms",
    ylabel="E2E Latency (ms)",
    title="E2EL Across Interconnects",
    labels=["Single GPU (TP=1)", "NVLink (TP=2)", "SHM (TP=2)", "Network Socket (TP=2)"],
    colors=["#4C9BE8", "#E85C4C", "#9B59B6", "#E8A84C"],
    subtitle_info=sub,
)

fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp2_shm_df, tp2_nwfb_df],
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across Interconnects",
    labels=["Single GPU (TP=1)", "NVLink (TP=2)", "SHM (TP=2)", "Network Socket (TP=2)"],
    colors=["#4C9BE8", "#E85C4C", "#9B59B6", "#E8A84C"],
    log_scale=True,
    subtitle_info=sub,
)

#%% ======================================================================
# all configs plotting
# ========================================================================
# across all configs
fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp2_shm_df, tp2_nwfb_df, tp4_df],
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across All Configs",
    labels=["Single GPU (TP=1)", "NVLink (TP=2)", "SHM (TP=2)", "Network Socket (TP=2)", "NVLink (TP=4)"],
    colors=["#4C9BE8", "#E85C4C", "#9B59B6", "#E8A84C", "#5CB85C"],
    subtitle_info=sub,
)
fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp2_shm_df, tp2_nwfb_df, tp4_df],
    metric="mean_ttft_ms",
    ylabel="TTFT (ms)",
    title="TTFT Across All Configs",
    labels=["Single GPU (TP=1)", "NVLink (TP=2)", "SHM (TP=2)", "Network Socket (TP=2)", "NVLink (TP=4)"],
    colors=["#4C9BE8", "#E85C4C", "#9B59B6", "#E8A84C", "#5CB85C"],
    subtitle_info=sub,
)
fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp2_shm_df, tp2_nwfb_df, tp4_df],
    metric="mean_e2el_ms",
    ylabel="E2EL (ms)",
    title="E2EL Across All Configs",
    labels=["Single GPU (TP=1)", "NVLink (TP=2)", "SHM (TP=2)", "Network Socket (TP=2)", "NVLink (TP=4)"],
    colors=["#4C9BE8", "#E85C4C", "#9B59B6", "#E8A84C", "#5CB85C"],
    subtitle_info=sub,
)
fig = plot_metric_vs_concurrency(
    [tp1_df, tp2_df, tp2_shm_df, tp2_nwfb_df, tp4_df],
    metric="output_throughput",
    ylabel="Throughput (tokens/s)",
    title="Throughput Across All Configs",
    labels=["Single GPU (TP=1)", "NVLink (TP=2)", "SHM (TP=2)", "Network Socket (TP=2)", "NVLink (TP=4)"],
    colors=["#4C9BE8", "#E85C4C", "#9B59B6", "#E8A84C", "#5CB85C"],
    subtitle_info=sub,
)

# %%
# ========================================================================
# derived metrics plotting
# ========================================================================

def plot_throughput_efficiency(
        df, title, labels, legend_labels,
        subtitle_info=None, 
        paper=False):
    
    fig, ax = plt.subplots(figsize=(4,3))
    colors = ["#E85C4C", "#9B59B6", "#E8A84C", "#5CB85C"]

    for label, legend_label, color in zip(labels, legend_labels, colors):
        filtered_df = filter_df(df, 'label', label)
        ax.plot(filtered_df['max_concurrency'], 
                filtered_df['throughput_efficiency'],
                color=color, label=legend_label)

    ax.axhline(1.0, color='#444444', linewidth=0.8, linestyle='--')
    ax.text(7, 1.02, "ideal (efficiency = 1)", size=6)
    ax.set_title(title, pad=25)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('Efficiency')
    ax.set_xticks(df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7, loc='best')
    ax.legend(loc='lower center', bbox_to_anchor=(1.02, 0.07), 
              title='Configuration')
    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)
    plt.tight_layout()
    return fig

def plot_relative_speedup(
    df, title, metric, 
    labels, legend_labels,
    subtitle_info=None, 
    paper=False):
    
    fig, ax = plt.subplots(figsize=(4.5,3))
    colors = ["#E85C4C", "#9B59B6", "#E8A84C", "#5CB85C"]

    for label, legend_label, color in zip(labels, legend_labels, colors):
        filtered_df = filter_df(df, 'label', label)
        ax.plot(filtered_df['max_concurrency'], 
                1/filtered_df[metric],
                label=legend_label, color=color)

    ax.axhline(1.0, color="#4C9BE8", linewidth=0.8, linestyle='--', label='Baseline (TP=1)')
    ax.set_title(title, pad=25)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('Speedup')
    ax.set_xticks(df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              title='Configuration')
    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)
    plt.tight_layout()
    return fig

def plot_speedup_all(
        df, labels, legend_labels,
        subtitle_info=None,
        paper=False):
    
    fig, axes = plt.subplots(1, 3, figsize=(10,3), sharex=True)

    colors = ["#E85C4C", "#9B59B6", "#E8A84C", "#5CB85C"]
    metrics = ['relative_ttft_slowdown', 'relative_itl_slowdown', 'relative_e2el_slowdown']
    titles = ['TTFT', 'ITL', 'E2EL']

    handles = []
    title = 'Speedup of Inference Metrics Across Configs'
    for ax, metric, title in zip(axes, metrics, titles):
        for label, legend_label, color in zip(labels, legend_labels, colors):
            filtered_df = filter_df(df, 'label', label)

            line, = ax.plot(filtered_df['max_concurrency'], 
                1/filtered_df[metric],
                label=legend_label, color=color)

            if len(handles) < len(labels):
                handles.append(line)

        line = ax.axhline(1.0, color="#4C9BE8", linewidth=0.8, linestyle='--', label='Baseline (TP=1)')
        handles.append(line)
        ax.set_title(title)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, axis='y', alpha=0.3)
    
    
    axes[0].set_ylabel('Speedup')
    for ax in axes:
        ax.set_xlabel('Concurrency')

    fig.legend(handles, legend_labels + ['Baseline (TP=1)'],
               loc='center left', bbox_to_anchor=(1, 0.5),
               title='Configuration',
               fontsize=7)

    fig.suptitle('Speedup of Inference Metrics Across Configs')
    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)

    plt.tight_layout()
    return fig
    

def plot_normalized_throughput(
        dfs, labels, colors, tps, title,
        subtitle_info=None, 
        paper=False):
    
    fig, ax = plt.subplots()

    for df, label, color, tp in zip(dfs, labels, colors, tps):
        ax.plot(df['max_concurrency'], df['output_throughput'] / tp, marker='o', 
                label=label, markersize=2, color=color)
 
    ax.set_title(title, pad=20)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('Throughput per GPU')
    ax.set_xticks(df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)
    plt.tight_layout()
    return fig

# for label, grp in derived.groupby('label'):
#     print(f"\n── {label} ──")
#     print(grp[['max_concurrency', 'relative_itl_slowdown',
#                 'degradation', 'observed_allreduce_latency_ms',
#                 'throughput_efficiency', 'observed_comm_overhead_ms']].to_string(index=False))
    
labels = ['mainsweep','p2pdisabled','networkfallback','mainsweep_tp4']
legend_labels = ['NVLink (TP=2)', 'P2P Disabled (TP=2)', 'Network Socket (TP=2)', 'NVLink (TP=4)']

fig = plot_throughput_efficiency(
    derived,
    title='Throughput Efficiency',
    labels=labels,
    legend_labels=legend_labels,
    subtitle_info=sub)

# fig = plot_relative_speedup(
#     derived,
#     title='ITL Speedup Across Configs',
#     metric='relative_itl_slowdown',
#     labels=labels,   
#     legend_labels=legend_labels, 
#     subtitle_info=sub
# )
# fig = plot_speedup(
#     derived,
#     metric='relative_ttft_slowdown',
#     title='TTFT Speedup Across Configs',
#     subtitle_info=sub
# )
# fig = plot_relative_speedup(
#     derived,
#     metric='relative_e2el_slowdown',
#     title='E2EL Speedup Across Configs',
#     labels=labels,  
#     legend_labels=legend_labels,
#     subtitle_info=sub
# )
fig = plot_normalized_throughput(
    [tp1_df, tp2_df, tp4_df],
    title="Per-GPU Normalized Output Throughput",
    labels=["TP=1", "TP=2", "TP=4"],
    colors=["#4C9BE8", "#E85C4C", "#5CB85C"],
    subtitle_info=sub,
    tps=[1,2,4]
)

fig = plot_speedup_all(
    derived, 
    labels=labels,
    legend_labels=legend_labels,
)
# %%
