#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

from utils.plotting import *
from utils.shared_config import *

set_figure_style()

# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

data_path = "../results/concurrency-sweep/averaged_concurrency_sweep_results.csv"
data = pd.read_csv(data_path)

derived_metrics_path = "../results/concurrency-sweep/derived_metrics.csv"
derived = pd.read_csv(derived_metrics_path)
derived = derived[derived['max_concurrency'] > 1]

# --------------------------------------------------------------------------
# plot functions
# --------------------------------------------------------------------------

def plot_metric_vs_concurrency(
    series, df, metric, ylabel, title,
    saturation_lines=None,
    log_scale=False,
    y_lim=None,
    subtitle_info=None,
    paper=False,
):
    """ line plot of metric vs concurrency for a list of series entries """
    fig, ax = plt.subplots()

    for s in series:
        # sort values before plotting
        plot_df = filter_df(df, **s['filters']).sort_values("max_concurrency")
        ax.plot(
            plot_df['max_concurrency'],
            plot_df[metric],
            marker=s['marker'],
            markersize=2,
            label=s["label"],
            color=s["color"],
            linestyle=s['linestyle'],
        )

    if y_lim != None:
        ax.set_ylim(y_lim[0], y_lim[1])

    if saturation_lines:
        for x_val, _ in saturation_lines:
            ax.axvline(x=x_val, color='grey', linewidth=0.9, linestyle=':')

    if log_scale:
        ax.set_yscale('log', base=2)
        ticks_s = [0.1, 1, 10, 60]
        ticks_ms = 1000 * np.array(ticks_s)
        ax.set_yticks(ticks_ms)
        ax.set_yticklabels([f"{t}s" for t in ticks_s])
 
    ax.set_title(title, pad=15)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_xticks(plot_df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)
    plt.tight_layout()
    return fig

def plot_throughput_efficiency(
        series, df, title,
        subtitle_info=None, 
        paper=False):
    """ throughput efficiency relative to baseline vs concurrency for a series """
    fig, ax = plt.subplots(figsize=(4,3))

    for s in series:
        plot_df = filter_df(df, **s['filters']).sort_values('max_concurrency')
        ax.plot(plot_df['max_concurrency'], 
                plot_df['throughput_efficiency'],
                color=s['color'], label=s['label'])

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

def plot_speedup_all(
        series, df, title,
        subtitle_info=None,
        paper=False):
    """ 3-panel speedup plot (TTFT / ITL / E2EL ) vs concurrency """
    fig, axes = plt.subplots(1, 3, figsize=(10,3), sharex=True)

    metrics = ['ttft_slowdown', 'itl_slowdown', 'e2el_slowdown']
    panel_titles = ['TTFT', 'ITL', 'E2EL']
    handles = []

    for ax, metric, panel_title in zip(axes, metrics, panel_titles):
        for s in series:
            plot_df = filter_df(df, **s['filters']).sort_values('max_concurrency')
            line, = ax.plot(plot_df['max_concurrency'], 1 / plot_df[metric],
                label=s['label'], color=s['color'])
            if len(handles) < len(series):
                handles.append(line)

        baseline_line = ax.axhline(1.0, color="#4C9BE8", linewidth=0.8, 
                                   linestyle='--', label='Baseline (TP=1)')
        ax.set_title(panel_title)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xlabel('Concurrency')
    
    handles.append(baseline_line)
    axes[0].set_ylabel('Speedup')

    fig.legend(handles, 
               [s['label'] for s in series] + ['Baseline (TP=1)'],
               loc='center left', bbox_to_anchor=(1, 0.5),
               title='Configuration', fontsize=7)

    fig.suptitle(title)
    if subtitle_info and not paper:
        subtitle(axes[-1], **subtitle_info)

    plt.tight_layout()
    return fig
    
def plot_normalized_throughput(
        series, df, title,
        subtitle_info=None, 
        paper=False):
    
    fig, ax = plt.subplots()

    for s in series:
        plot_df = filter_df(df, **s['filters']).sort_values("max_concurrency")

        x_vals = plot_df['max_concurrency'] / plot_df["tp"]
        y_vals = plot_df['output_throughput'] / plot_df["tp"]

        mask = (x_vals >= 4) & (x_vals<=512)

        ax.plot(
            x_vals[mask], 
            y_vals[mask], 
            marker=s["marker"], label=s["label"], 
            markersize=2, color=s["color"], linestyle=s["linestyle"])
 
    ax.set_title(title, pad=15)
    ax.set_xlabel('Concurrency per GPU (requests)')
    ax.set_ylabel('Throughput per GPU (token/s)')
    ax.set_xticks(plot_df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)
    plt.tight_layout()
    return fig

# --------------------------------------------------------------------------
# TP SCALING
# --------------------------------------------------------------------------

#%% 
# A100 TP scaling (primary model)

sub_a100_8b = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')
a100_tp_8b = build_tp_series('a100', 'llama-3.1-8b')


fig_itl = plot_metric_vs_concurrency(
    a100_tp_8b, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_8b
)
fig_thp = plot_metric_vs_concurrency(
    a100_tp_8b, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_a100_8b
)
fig_ttft = plot_metric_vs_concurrency(
    a100_tp_8b, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_a100_8b
)
fig_e2el = plot_metric_vs_concurrency(
    a100_tp_8b, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_8b
)

fig = plot_throughput_efficiency(
    a100_tp_8b,
    derived,
    title="Throughput Efficiency Across TP Degrees (A100)",
    subtitle_info=sub_a100_8b
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_tp_8b, data,
    title="Primary Model: Normalized Output Throughput",
    subtitle_info=sub_a100_8b,
)

fig = plot_speedup_all(a100_tp_8b, derived, 'Speedup Across TP Degree (A100)')


#%% 
# A100 TP scaling (1b model)

sub_a100_1b = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.2-1B', 
           config='ISL/OSL = 512/512')
a100_tp_1b = build_tp_series('a100', 'llama-3.2-1b')


fig_itl = plot_metric_vs_concurrency(
    a100_tp_1b, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_1b
)
fig_thp = plot_metric_vs_concurrency(
    a100_tp_1b, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_a100_1b
)
fig_ttft = plot_metric_vs_concurrency(
    a100_tp_1b, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_a100_1b
)
fig_e2el = plot_metric_vs_concurrency(
    a100_tp_1b, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_1b
)

fig = plot_throughput_efficiency(
    a100_tp_1b,
    derived,
    title="Throughput Efficiency Across TP Degrees (A100)",
    subtitle_info=sub_a100_1b
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_tp_1b, data,
    title="Smaller Model: Normalized Output Throughput",
    subtitle_info=sub_a100_1b
)

fig = plot_speedup_all(a100_tp_1b, derived, 'Speedup Across TP Degree (A100)')
#%% 
# A100 TP scaling (13b model)

sub_a100_13b = dict(gpu='shy-fec (A100s)', 
           model='Llama-2-13B', 
           config='ISL/OSL = 512/512')
a100_tp_13b = build_tp_series('a100', 'llama-2-13b')


fig_itl = plot_metric_vs_concurrency(
    a100_tp_13b, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_13b
)
fig_thp = plot_metric_vs_concurrency(
    a100_tp_13b, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_a100_13b
)
fig_ttft = plot_metric_vs_concurrency(
    a100_tp_13b, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_a100_13b
)
fig_e2el = plot_metric_vs_concurrency(
    a100_tp_13b, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_13b
)

fig = plot_throughput_efficiency(
    a100_tp_13b,
    derived,
    title="Throughput Efficiency Across TP Degrees (A100)",
    subtitle_info=sub_a100_13b
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_tp_13b, data,
    title="Larger Model: Normalized Output Throughput",
    subtitle_info=sub_a100_13b,
)

fig = plot_speedup_all(a100_tp_13b, derived, 'Speedup Across TP Degree (A100)')

#%% 
# V100 TP scaling(primary model)

sub_v100_8b = dict(gpu='funnotch (V100s)', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')
v100_tp_8b = build_tp_series('v100', 'llama-3.1-8b')


fig_itl = plot_metric_vs_concurrency(
    v100_tp_8b, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_v100_8b
)
fig_thp = plot_metric_vs_concurrency(
    v100_tp_8b, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_v100_8b
)
fig_ttft = plot_metric_vs_concurrency(
    v100_tp_8b, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_v100_8b
)
fig_e2el = plot_metric_vs_concurrency(
    v100_tp_8b, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_v100_8b
)

fig = plot_throughput_efficiency(
    v100_tp_8b,
    derived,
    title="Throughput Efficiency Across TP Degrees (V100)",
    subtitle_info=sub_v100_8b
)

fig_v100_thp_normalized = plot_normalized_throughput(
    v100_tp_8b, data,
    title="Primary Model: Normalized Output Throughput",
    subtitle_info=sub_v100_8b,
)

fig = plot_speedup_all(v100_tp_8b, derived, 'Speedup Across TP Degree (V100)')

#%%
# V100 TP scaling (1b model) 

sub_v100_1b = dict(gpu='funnotch (V100s)', 
           model='Llama-3.2-1B', 
           config='ISL/OSL = 512/512')
v100_tp_1b = build_tp_series('v100', 'llama-3.2-1b')

fig_itl = plot_metric_vs_concurrency(
    v100_tp_1b, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_v100_1b
)
fig_thp = plot_metric_vs_concurrency(
    v100_tp_1b, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_v100_1b
)
fig_ttft = plot_metric_vs_concurrency(
    v100_tp_1b, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_v100_1b
)
fig_e2el = plot_metric_vs_concurrency(
    v100_tp_1b, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_v100_1b
)

fig = plot_throughput_efficiency(
    v100_tp_1b,
    derived,
    title="Throughput Efficiency Across TP Degrees (V100)",
    subtitle_info=sub_v100_1b
)

fig_v100_thp_normalized = plot_normalized_throughput(
    v100_tp_1b, data,
    title="Smaller Model: Normalized Output Throughput",
    subtitle_info=sub_v100_1b,
)

fig = plot_speedup_all(v100_tp_1b, derived, 'Speedup Across TP Degree (V100)')

#%%
# V100 TP scaling (13b model) 

sub_v100_13b = dict(gpu='funnotch (V100s)', 
           model='Llama-2-13B', 
           config='ISL/OSL = 512/512')
v100_tp_13b = build_tp_series('v100', 'llama-2-13b')

fig_itl = plot_metric_vs_concurrency(
    v100_tp_13b, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_v100_13b
)
fig_thp = plot_metric_vs_concurrency(
    v100_tp_13b, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_v100_13b
)
fig_ttft = plot_metric_vs_concurrency(
    v100_tp_13b, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_v100_13b
)
fig_e2el = plot_metric_vs_concurrency(
    v100_tp_13b, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_v100_13b
)

fig = plot_throughput_efficiency(
    v100_tp_13b,
    derived,
    title="Throughput Efficiency Across TP Degrees (V100)",
    subtitle_info=sub_v100_13b
)

fig_v100_thp_normalized = plot_normalized_throughput(
    v100_tp_13b, data,
    title="Larger Model: Normalized Output Throughput",
    subtitle_info=sub_v100_13b,
)

fig = plot_speedup_all(v100_tp_13b, derived, 'Speedup Across TP Degree (V100)')


# --------------------------------------------------------------------------
# INTERCONNECT COMPARISON
# --------------------------------------------------------------------------

#%% 
# A100 interconnect comparison

a100_interconnect_tp2 = build_interconnect_series('a100', 'llama-3.1-8b', tp=2)

fig_thp = plot_metric_vs_concurrency(
    a100_interconnect_tp2, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across Interconnects",
    subtitle_info=sub_a100_8b,
)

fig_itl = plot_metric_vs_concurrency(
    a100_interconnect_tp2, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Interconnects",
    subtitle_info=sub_a100_8b,
)

fig_e2el = plot_metric_vs_concurrency(
    a100_interconnect_tp2, data,
    metric="mean_e2el_ms",
    ylabel="E2E Latency (ms)",
    title="E2EL Across Interconnects",
    subtitle_info=sub_a100_8b,
)

fig_ttft = plot_metric_vs_concurrency(
    a100_interconnect_tp2, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across Interconnects",
    subtitle_info=sub_a100_8b,
)

fig = plot_throughput_efficiency(
    a100_interconnect_tp2,
    derived,
    title="Throughput Efficiency Across Interconnects (A100)",
    subtitle_info=sub_a100_8b
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_interconnect_tp2, data,
    title="Normalized Output Throughput",
    subtitle_info=sub_a100_8b,
    paper=True,
)

fig = plot_speedup_all(a100_interconnect_tp2, derived, 'Speedup Across Interconnects (A100)')

#%% 
# A100 interconnect comparison tp4

a100_interconnect_tp4 = build_interconnect_series('a100', 'llama-3.1-8b', tp=4)

fig_thp = plot_metric_vs_concurrency(
    a100_interconnect_tp4 , data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across Interconnects",
    subtitle_info=sub_a100_8b,
)

fig_itl = plot_metric_vs_concurrency(
    a100_interconnect_tp4 , data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Interconnects",
    subtitle_info=sub_a100_8b,
)

fig_e2el = plot_metric_vs_concurrency(
    a100_interconnect_tp4 , data,
    metric="mean_e2el_ms",
    ylabel="E2E Latency (ms)",
    title="E2EL Across Interconnects",
    subtitle_info=sub_a100_8b,
)

fig_ttft = plot_metric_vs_concurrency(
    a100_interconnect_tp4 , data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across Interconnects",
    subtitle_info=sub_a100_8b,
)

fig = plot_throughput_efficiency(
    a100_interconnect_tp4 ,
    derived,
    title="Throughput Efficiency Across Interconnects (A100)",
    subtitle_info=sub_a100_8b
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_interconnect_tp4, data,
    title="Normalized Output Throughput",
    subtitle_info=sub_a100_8b,
    paper=True,
)

fig = plot_speedup_all(a100_interconnect_tp4 , derived, 'Speedup Across Interconnects (A100)')

#%% 
# V100 interconnect comparison

v100_interconnect_tp2 = build_interconnect_series('v100', 'llama-3.1-8b', tp=2)

fig_thp = plot_metric_vs_concurrency(
    v100_interconnect_tp2, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across Interconnects",
    subtitle_info=sub_v100_8b,
)

fig_itl = plot_metric_vs_concurrency(
    v100_interconnect_tp2, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Interconnects",
    subtitle_info=sub_v100_8b,
)

fig_e2el = plot_metric_vs_concurrency(
    v100_interconnect_tp2, data,
    metric="mean_e2el_ms",
    ylabel="E2E Latency (ms)",
    title="E2EL Across Interconnects",
    subtitle_info=sub_v100_8b,
)

fig_ttft = plot_metric_vs_concurrency(
    v100_interconnect_tp2, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across Interconnects",
    subtitle_info=sub_v100_8b,
)

fig = plot_throughput_efficiency(
    v100_interconnect_tp2,
    derived,
    title="Throughput Efficiency Across Interconnects (V100)",
    subtitle_info=sub_v100_8b
)

fig_v100_thp_normalized = plot_normalized_throughput(
    v100_interconnect_tp2, data,
    title="Normalized Output Throughput",
    subtitle_info=sub_v100_8b,
    paper=True,
)

fig = plot_speedup_all(v100_interconnect_tp2, derived, 'Speedup Across Interconnects (V100)')


#%% 
# V100 interconnect comparison tp=4

v100_interconnect_tp4 = build_interconnect_series('v100', 'llama-3.1-8b', tp=4)

fig_thp = plot_metric_vs_concurrency(
    v100_interconnect_tp4, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across Interconnects",
    subtitle_info=sub_v100_8b,
)

fig_itl = plot_metric_vs_concurrency(
    v100_interconnect_tp4, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Interconnects",
    subtitle_info=sub_v100_8b,
)

fig_e2el = plot_metric_vs_concurrency(
    v100_interconnect_tp4, data,
    metric="mean_e2el_ms",
    ylabel="E2E Latency (ms)",
    title="E2EL Across Interconnects",
    subtitle_info=sub_v100_8b,
)

fig_ttft = plot_metric_vs_concurrency(
    v100_interconnect_tp4, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across Interconnects",
    subtitle_info=sub_v100_8b,
)

fig = plot_throughput_efficiency(
    v100_interconnect_tp4,
    derived,
    title="Throughput Efficiency Across Interconnects (V100)",
    subtitle_info=sub_v100_8b
)

fig_v100_thp_normalized = plot_normalized_throughput(
    v100_interconnect_tp4, data,
    title="Normalized Output Throughput",
    subtitle_info=sub_v100_8b,
    paper=True,
)

fig = plot_speedup_all(v100_interconnect_tp4, derived, 'Speedup Across Interconnects (V100)')



# %%

def plot_throughput_efficiency(
        series, df, title,
        subtitle_info=None, 
        paper=False):
    """ throughput efficiency relative to baseline vs concurrency for a series """
    fig, ax = plt.subplots(figsize=(4,3), layout="constrained")

    for s in series:
        plot_df = filter_df(df, **s['filters']).sort_values('concurrency_per_gpu')
        
        x_vals = plot_df['concurrency_per_gpu']
        y_vals = plot_df['throughput_efficiency']

        mask = (x_vals >= 4) & (x_vals <= 512)

        ax.plot(x_vals[mask], 
                y_vals[mask],
                color=s['color'], label=s['label'], linestyle=s['linestyle'])

    ax.axhline(1.0, color='#444444', linewidth=0.8, linestyle='--')
    ax.text(7, 1.02, "ideal (efficiency = 1)", size=6)
    ax.set_title(title, pad=15)
    ax.set_xlabel('Concurrency per GPU')
    ax.set_ylabel('Efficiency')
    ax.set_xticks(df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)


    # color legend
    color_handles = [
        Line2D([0], [0], color='#E85C4C', lw=2, label='Default'),
        Line2D([0], [0], color='#9B59B6', lw=2, label='SHM'),
        Line2D([0], [0], color='#E8A84C', lw=2, label='Socket'),
    ]

    # Linestyle legend
    style_handles = [
        Line2D(
            [0], [0],
            color='black',
            lw=2,
            linestyle='-',
            marker='o',
            markersize=2,
            label='TP=2'
        ),
        Line2D(
            [0], [0],
            color='black',
            lw=2,
            linestyle='--',
            marker='o',
            markersize=2,
            label='TP=4'
        ),
    ]

    legend1 = ax.legend(
        handles=color_handles,
        title="Transport",
        fontsize=6,
        title_fontsize=7,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0)
    )

    legend2 = ax.legend(
        handles=style_handles,
        title="TP degree",
        title_fontsize=7,
        fontsize=6,
        loc='upper left',
        bbox_to_anchor=(1.02, 0.7)
    )

    ax.add_artist(legend1)

    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)
    return fig

def plot_normalized_throughput(
        series, df, title,
        subtitle_info=None,
        paper=False):

    fig, ax = plt.subplots(layout="constrained")

    # plot data
    for s in series:
        plot_df = filter_df(df, **s['filters']).sort_values("max_concurrency")

        x_vals = plot_df['max_concurrency'] / plot_df["tp"]
        y_vals = plot_df['output_throughput'] / plot_df["tp"]

        mask = (x_vals >= 4) & (x_vals <= 512)

        ax.plot(
            x_vals[mask],
            y_vals[mask],
            marker=s["marker"],
            markersize=2,
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=1.5
        )


    # axis formatting
    ax.set_title(title, pad=20)
    ax.set_xlabel('Concurrency per GPU (requests)')
    ax.set_ylabel('Throughput per GPU (token/s)')

    ax.set_xticks(plot_df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.grid(True, axis='y', alpha=0.3)

    # color legend
    color_handles = [
        Line2D([0], [0], color='#4C9BE8', lw=1, label='Single-GPU'),
        Line2D([0], [0], color='#E85C4C', lw=2, label='Default'),
        Line2D([0], [0], color='#9B59B6', lw=2, label='SHM'),
        Line2D([0], [0], color='#E8A84C', lw=2, label='Socket'),
    ]

    # Linestyle legend
    style_handles = [
        Line2D(
            [0], [0],
            color='black',
            lw=2,
            linestyle='-',
            marker='o',
            markersize=2,
            label='TP=2'
        ),
        Line2D(
            [0], [0],
            color='black',
            lw=2,
            linestyle='--',
            marker='o',
            markersize=2,
            label='TP=4'
        ),
    ]

    legend1 = ax.legend(
        handles=color_handles,
        title="Transport",
        fontsize=6,
        title_fontsize=7,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0)
    )

    legend2 = ax.legend(
        handles=style_handles,
        title="TP degree",
        title_fontsize=7,
        fontsize=6,
        loc='upper left',
        bbox_to_anchor=(1.02, 0.55)
    )

    ax.add_artist(legend1)

    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)

    #plt.tight_layout(rect=[0,0,0.8,1])

    return fig


for config in a100_interconnect_tp4:
    config["linestyle"] = "--"
for config in v100_interconnect_tp4:
    config["linestyle"] = "--"
a100_interconnect = a100_interconnect_tp2 + a100_interconnect_tp4
v100_interconnect = v100_interconnect_tp2 + v100_interconnect_tp4


sub_v100_8b = dict(gpu='V100 GPUs', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')
sub_a100_8b = dict(gpu='A100 GPUs', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')


# fig_v100_thp_normalized = plot_normalized_throughput(
#     v100_interconnect, data,
#     title="Normalized Output Throughput",
#     subtitle_info=sub_v100_8b,
#     paper=False,
# )

# fig_a100_thp_normalized = plot_normalized_throughput(
#     a100_interconnect, data,
#     title="Normalized Output Throughput",
#     subtitle_info=sub_a100_8b,
# )

#%%


for i, s in enumerate(a100_interconnect):
    print(f"{i} {s}")


idx = [1,2,3,5,6,]
v100_subset = [v100_interconnect[i] for i in idx]
a100_subset = [a100_interconnect[i] for i in idx]

fig = plot_throughput_efficiency(
    v100_subset,
    derived,
    title="Funnotch: Throughput Efficiency Across Transports",
    subtitle_info=sub_v100_8b
)


fig = plot_throughput_efficiency(
    a100_subset,
    derived,
    title="Shy-fec: Throughput Efficiency Across Transports",
    subtitle_info=sub_a100_8b
)



# %%
