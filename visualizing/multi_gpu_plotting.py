#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from plot_utils import *


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

data_path = "../results/concurrency-sweep/averaged_concurrency_sweep_results.csv"
data = pd.read_csv(data_path)

derived_metrics_path = "../results/concurrency-sweep/derived_metrics.csv"
derived = pd.read_csv(derived_metrics_path)
derived = derived[derived['max_concurrency'] > 1]

# --------------------------------------------------------------------------
# set plotting style and colors
# --------------------------------------------------------------------------

set_figure_style()

TP_COLORS = {
    1: "#4C9BE8", 
    2:"#E85C4C", 
    4: "#5CB85C"}
INTERCONNECT_COLORS = {
    "default": "#E85C4C",
    "p2p_disabled":"#9B59B6",
    "network_socket": "#E8A84C" 
}

# --------------------------------------------------------------------------
# series for data organization
# --------------------------------------------------------------------------

def series_entry(label, color, linestyle='-', marker='o', **filters):
    """ data series defined by its filters """
    return {
        "filters": filters,
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
    }

def build_tp_series(gpu_type, model_name):
    """ TP=1,2,4 series using default interconnect """
    series = []
    for tp in [1,2,4]:
        series.append(series_entry(
            label=f"TP={tp}",
            color=TP_COLORS[tp],
            gpu_type=gpu_type,
            model_name=model_name,
            interconnect='default',
            tp=tp
        ))
    return series

def build_interconnect_series(gpu_type, model_name, tp=2):
    """ interconnect series including TP=1 baseline  """
    default_label = "NVLink" if gpu_type == "a100" else "PCIe"
    series = []
    series.append(series_entry(
        label=f"Single-GPU (TP=1)",
        color=TP_COLORS[1],
        gpu_type=gpu_type,
        model_name=model_name,
        interconnect='default',
        tp=1
    ))
    series.append(series_entry(
        label=f"{default_label} (TP={tp})",
        color=INTERCONNECT_COLORS['default'],
        gpu_type=gpu_type,
        model_name=model_name,
        interconnect='default',
        tp=tp
    ))
    series.append(series_entry(
        label=f"SHM (TP={tp})",
        color=INTERCONNECT_COLORS['p2p_disabled'],
        gpu_type=gpu_type,
        model_name=model_name,
        interconnect='p2p_disabled',
        tp=tp
    ))
    series.append(series_entry(
        label=f"Socket (TP={tp})",
        color=INTERCONNECT_COLORS['network_socket'],
        gpu_type=gpu_type,
        model_name=model_name,
        interconnect='network_socket',
        tp=tp
    ))
    return series

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
            markersize=2, color=s["color"])
 
    ax.set_title(title, pad=20)
    ax.set_xlabel('Concurrency')
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

sub_a100 = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')
a100_tp = build_tp_series('a100', 'llama-3.1-8b')


fig_itl = plot_metric_vs_concurrency(
    a100_tp, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100
)
fig_thp = plot_metric_vs_concurrency(
    a100_tp, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_a100
)
fig_ttft = plot_metric_vs_concurrency(
    a100_tp, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_a100
)
fig_e2el = plot_metric_vs_concurrency(
    a100_tp, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100
)

fig = plot_throughput_efficiency(
    a100_tp,
    derived,
    title="Throughput Efficiency Across TP Degrees (A100)",
    subtitle_info=sub_a100
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_tp, data,
    title="Shy-fec (A100s): Per-GPU Output Throughput",
    subtitle_info=sub_a100,
    paper=True,
)

fig = plot_speedup_all(a100_tp, derived, 'Speedup Across TP Degree (A100)')

#%% 
# A100 TP scaling (13b model)

sub_a100 = dict(gpu='shy-fec (A100s)', 
           model='Llama-2-13B', 
           config='ISL/OSL = 512/512')
a100_tp = build_tp_series('a100', 'llama-2-13b')


fig_itl = plot_metric_vs_concurrency(
    a100_tp, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100
)
fig_thp = plot_metric_vs_concurrency(
    a100_tp, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_a100
)
fig_ttft = plot_metric_vs_concurrency(
    a100_tp, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_a100
)
fig_e2el = plot_metric_vs_concurrency(
    a100_tp, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100
)

fig = plot_throughput_efficiency(
    a100_tp,
    derived,
    title="Throughput Efficiency Across TP Degrees (A100)",
    subtitle_info=sub_a100
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_tp, data,
    title="Shy-fec (A100s): Per-GPU Output Throughput",
    subtitle_info=sub_a100,
    paper=True,
)

fig = plot_speedup_all(a100_tp, derived, 'Speedup Across TP Degree (A100)')

#%% 
# V100 TP scaling(primary model)

sub_v100 = dict(gpu='funnotch (V100s)', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')
v100_tp = build_tp_series('v100', 'llama-3.1-8b')


fig_itl = plot_metric_vs_concurrency(
    v100_tp, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_v100
)
fig_thp = plot_metric_vs_concurrency(
    v100_tp, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_v100
)
fig_ttft = plot_metric_vs_concurrency(
    v100_tp, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_v100
)
fig_e2el = plot_metric_vs_concurrency(
    v100_tp, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_v100
)

fig = plot_throughput_efficiency(
    v100_tp,
    derived,
    title="Throughput Efficiency Across TP Degrees (V100)",
    subtitle_info=sub_v100
)

fig_v100_thp_normalized = plot_normalized_throughput(
    v100_tp, data,
    title="Funnotch (V100s): Per-GPU Output Throughput",
    subtitle_info=sub_v100,
    paper=True,
)

fig = plot_speedup_all(v100_tp, derived, 'Speedup Across TP Degree (V100)')

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
    title="Funnotch (V100s): Per-GPU Output Throughput",
    subtitle_info=sub_v100_1b,
    paper=True,
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
    title="Funnotch (V100s): Per-GPU Output Throughput",
    subtitle_info=sub_v100_13b,
    paper=True,
)

fig = plot_speedup_all(v100_tp_13b, derived, 'Speedup Across TP Degree (V100)')


# --------------------------------------------------------------------------
# INTERCONNECT COMPARISON
# --------------------------------------------------------------------------

#%% 
# A100 interconnect comparison

a100_interconnect = build_interconnect_series('a100', 'llama-3.1-8b')

fig_thp = plot_metric_vs_concurrency(
    a100_interconnect, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across Interconnects",
    subtitle_info=sub_a100,
)

fig_itl = plot_metric_vs_concurrency(
    a100_interconnect, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Interconnects",
    subtitle_info=sub_a100,
)

fig_e2el = plot_metric_vs_concurrency(
    a100_interconnect, data,
    metric="mean_e2el_ms",
    ylabel="E2E Latency (ms)",
    title="E2EL Across Interconnects",
    subtitle_info=sub_a100,
)

fig_ttft = plot_metric_vs_concurrency(
    a100_interconnect, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across Interconnects",
    subtitle_info=sub_a100,
)

fig = plot_throughput_efficiency(
    a100_interconnect,
    derived,
    title="Throughput Efficiency Across Interconnects (A100)",
    subtitle_info=sub_a100
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_interconnect, data,
    title="Shy-fec (A100s): Per-GPU Output Throughput",
    subtitle_info=sub_a100,
    paper=True,
)

fig = plot_speedup_all(a100_interconnect, derived, 'Speedup Across Interconnects (A100)')

#%% 
# A100 interconnect comparison tp4

a100_interconnect_tp4 = build_interconnect_series('a100', 'llama-3.1-8b', tp=4)

fig_thp = plot_metric_vs_concurrency(
    a100_interconnect_tp4 , data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across Interconnects",
    subtitle_info=sub_a100,
)

fig_itl = plot_metric_vs_concurrency(
    a100_interconnect_tp4 , data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Interconnects",
    subtitle_info=sub_a100,
)

fig_e2el = plot_metric_vs_concurrency(
    a100_interconnect_tp4 , data,
    metric="mean_e2el_ms",
    ylabel="E2E Latency (ms)",
    title="E2EL Across Interconnects",
    subtitle_info=sub_a100,
)

fig_ttft = plot_metric_vs_concurrency(
    a100_interconnect_tp4 , data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across Interconnects",
    subtitle_info=sub_a100,
)

fig = plot_throughput_efficiency(
    a100_interconnect_tp4 ,
    derived,
    title="Throughput Efficiency Across Interconnects (A100)",
    subtitle_info=sub_a100
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_interconnect_tp4, data,
    title="Shy-fec (A100s): Per-GPU Output Throughput",
    subtitle_info=sub_a100,
    paper=True,
)

fig = plot_speedup_all(a100_interconnect_tp4 , derived, 'Speedup Across Interconnects (A100)')

#%% 
# V100 interconnect comparison

v100_interconnect = build_interconnect_series('v100', 'llama-3.1-8b')

fig_thp = plot_metric_vs_concurrency(
    v100_interconnect, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across Interconnects",
    subtitle_info=sub_v100,
)

fig_itl = plot_metric_vs_concurrency(
    v100_interconnect, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Interconnects",
    subtitle_info=sub_v100,
)

fig_e2el = plot_metric_vs_concurrency(
    v100_interconnect, data,
    metric="mean_e2el_ms",
    ylabel="E2E Latency (ms)",
    title="E2EL Across Interconnects",
    subtitle_info=sub_v100,
)

fig_ttft = plot_metric_vs_concurrency(
    v100_interconnect, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across Interconnects",
    subtitle_info=sub_v100,
)

fig = plot_throughput_efficiency(
    v100_interconnect,
    derived,
    title="Throughput Efficiency Across Interconnects (V100)",
    subtitle_info=sub_v100
)

fig_v100_thp_normalized = plot_normalized_throughput(
    v100_interconnect, data,
    title="Funnotch (V100s): Per-GPU Output Throughput",
    subtitle_info=sub_v100,
    paper=True,
)

fig = plot_speedup_all(v100_interconnect, derived, 'Speedup Across Interconnects (V100)')


#%% 
# V100 interconnect comparison tp=4

v100_interconnect_tp4 = build_interconnect_series('v100', 'llama-3.1-8b', tp=4)

fig_thp = plot_metric_vs_concurrency(
    v100_interconnect_tp4, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across Interconnects",
    subtitle_info=sub_v100,
)

fig_itl = plot_metric_vs_concurrency(
    v100_interconnect_tp4, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Interconnects",
    subtitle_info=sub_v100,
)

fig_e2el = plot_metric_vs_concurrency(
    v100_interconnect_tp4, data,
    metric="mean_e2el_ms",
    ylabel="E2E Latency (ms)",
    title="E2EL Across Interconnects",
    subtitle_info=sub_v100,
)

fig_ttft = plot_metric_vs_concurrency(
    v100_interconnect_tp4, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across Interconnects",
    subtitle_info=sub_v100,
)

fig = plot_throughput_efficiency(
    v100_interconnect_tp4,
    derived,
    title="Throughput Efficiency Across Interconnects (V100)",
    subtitle_info=sub_v100
)

fig_v100_thp_normalized = plot_normalized_throughput(
    v100_interconnect_tp4, data,
    title="Funnotch (V100s): Per-GPU Output Throughput",
    subtitle_info=sub_v100,
    paper=True,
)

fig = plot_speedup_all(v100_interconnect_tp4, derived, 'Speedup Across Interconnects (V100)')



# %%
