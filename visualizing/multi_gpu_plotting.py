#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import matplot2tikz

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

slo_data_path = "../results/concurrency-sweep/slo_attainment.csv"
slo_data = pd.read_csv(slo_data_path)
slo_cols = {
        'responsive': 'pergpu_limit_responsive', 'acceptable': 'pergpu_limit_acceptable'
    }
slo_styles = {
    'responsive': ('*', 60, '--'), 'acceptable': ('D', 30, ':'),
}


# --------------------------------------------------------------------------
# plot functions
# --------------------------------------------------------------------------

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
    ylabel="Mean ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_8b,
    paper=True
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
    log_scale=True,
    subtitle_info=sub_a100_8b,
    paper=True
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
    title="Normalized Output Throughput",
    subtitle_info=sub_a100_8b,
    paper=False
    #slo_df=slo_data, slo_cols=slo_cols, slo_styles=slo_styles
)

fig = plot_speedup_all(a100_tp_8b, derived, 'Speedup Across TP Degree (A100)')

#%% 
# A100 TP scaling when P2P Disabled (8b model)

a100_tp_8b_p2pdisabled = build_tp_series('a100', 'llama-3.1-8b', interconnect = 'p2p_disabled')

fig_itl = plot_metric_vs_concurrency(
    a100_tp_8b_p2pdisabled, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_8b
)
# fig_thp = plot_metric_vs_concurrency(
#     a100_tp_8b_p2pdisabled, data,
#     metric="output_throughput",
#     ylabel="Output Throughput (tokens/s)",
#     title="Output Throughput Across TP Configs",
#     subtitle_info=sub_a100_8b
# )
fig_ttft = plot_metric_vs_concurrency(
    a100_tp_8b_p2pdisabled, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_a100_8b
)
fig_e2el = plot_metric_vs_concurrency(
    a100_tp_8b_p2pdisabled, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_8b
)

fig = plot_throughput_efficiency(
    a100_tp_8b_p2pdisabled,
    derived,
    title="Throughput Efficiency Across TP Degrees (A100)",
    subtitle_info=sub_a100_8b
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_tp_8b_p2pdisabled, data,
    title="Primary Model with P2P Disabled: Throughput",
    subtitle_info=sub_a100_8b,
    slo_df=slo_data, slo_cols=slo_cols, slo_styles=slo_styles
)

fig = plot_speedup_all(a100_tp_8b_p2pdisabled, derived, 'Speedup Across TP Degree (A100)')


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
    subtitle_info=sub_a100_1b,
    #slo_df=slo_data, slo_cols=slo_cols, slo_styles=slo_styles
    paper=True
)

fig = plot_speedup_all(a100_tp_1b, derived, 'Speedup Across TP Degree (A100)')


#%% 
# A100 TP scaling when p2p disabled (1b model)

a100_tp_1b_p2pdisabled = build_tp_series('a100', 'llama-3.2-1b', interconnect='p2p_disabled')


fig_itl = plot_metric_vs_concurrency(
    a100_tp_1b_p2pdisabled, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_1b
)
fig_thp = plot_metric_vs_concurrency(
    a100_tp_1b_p2pdisabled, data,
    metric="output_throughput",
    ylabel="Output Throughput (tokens/s)",
    title="Output Throughput Across TP Configs",
    subtitle_info=sub_a100_1b
)
fig_ttft = plot_metric_vs_concurrency(
    a100_tp_1b_p2pdisabled, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_a100_1b
)
fig_e2el = plot_metric_vs_concurrency(
    a100_tp_1b_p2pdisabled, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_1b
)

fig = plot_throughput_efficiency(
    a100_tp_1b_p2pdisabled,
    derived,
    title="Throughput Efficiency Across TP Degrees (A100)",
    subtitle_info=sub_a100_1b
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_tp_1b_p2pdisabled, data,
    title="Smaller Model: Normalized Output Throughput",
    subtitle_info=sub_a100_1b,
    slo_df=slo_data, slo_cols=slo_cols, slo_styles=slo_styles
)

fig = plot_speedup_all(a100_tp_1b_p2pdisabled, derived, 'Speedup Across TP Degree (A100)')

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
    #slo_df=slo_data, slo_cols=slo_cols, slo_styles=slo_styles,
    paper=True
)

fig = plot_speedup_all(a100_tp_13b, derived, 'Speedup Across TP Degree (A100)')

#%% 
# A100 TP scaling when P2P Disabled (13b model)

sub_a100_13b = dict(gpu='shy-fec (A100s)', 
           model='Llama-2-13B', 
           config='ISL/OSL = 512/512')
a100_tp_13b_p2pdisabled = build_tp_series('a100', 'llama-2-13b', interconnect = 'p2p_disabled')

fig_itl = plot_metric_vs_concurrency(
    a100_tp_13b_p2pdisabled, data,
    metric="mean_itl_ms",
    ylabel="ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_13b
)
# fig_thp = plot_metric_vs_concurrency(
#     a100_tp_13b_p2pdisabled, data,
#     metric="output_throughput",
#     ylabel="Output Throughput (tokens/s)",
#     title="Output Throughput Across TP Configs",
#     subtitle_info=sub_a100_13b
# )
fig_ttft = plot_metric_vs_concurrency(
    a100_tp_13b_p2pdisabled, data,
    metric="mean_ttft_ms",
    ylabel="TTFT",
    title="TTFT Across TP Configs",
    log_scale=False,
    subtitle_info=sub_a100_13b
)
fig_e2el = plot_metric_vs_concurrency(
    a100_tp_13b_p2pdisabled, data,
    metric="mean_e2el_ms",
    ylabel="E2EL(ms)",
    title="E2EL Across Tensor Parallelism Configs",
    subtitle_info=sub_a100_13b
)

fig = plot_throughput_efficiency(
    a100_tp_13b_p2pdisabled,
    derived,
    title="Throughput Efficiency Across TP Degrees (A100)",
    subtitle_info=sub_a100_13b
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_tp_13b_p2pdisabled, data,
    title="Larger Model with P2P Disabled: Throughput",
    subtitle_info=sub_a100_13b,
    slo_df=slo_data, slo_cols=slo_cols, slo_styles=slo_styles
)

fig = plot_speedup_all(a100_tp_13b_p2pdisabled, derived, 'Speedup Across TP Degree (A100)')

#%% 
# V100 TP scaling(primary model)

sub_v100_8b = dict(gpu='funnotch (V100s)', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')
v100_tp_8b = build_tp_series('v100', 'llama-3.1-8b')


fig_itl = plot_metric_vs_concurrency(
    v100_tp_8b, data,
    metric="mean_itl_ms",
    ylabel="Mean ITL (ms)",
    title="ITL Across Tensor Parallelism Configs",
    subtitle_info=sub_v100_8b,
    paper=True
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
    log_scale=True,
    subtitle_info=sub_v100_8b,
    paper=True
)
# fig_e2el = plot_metric_vs_concurrency(
#     v100_tp_8b, data,
#     metric="mean_e2el_ms",
#     ylabel="E2EL(ms)",
#     title="E2EL Across Tensor Parallelism Configs",
#     subtitle_info=sub_v100_8b
# )

# fig = plot_throughput_efficiency(
#     v100_tp_8b,
#     derived,
#     title="Throughput Efficiency Across TP Degrees (V100)",
#     subtitle_info=sub_v100_8b
# )

fig_v100_thp_normalized = plot_normalized_throughput(
    v100_tp_8b, data,
    title="Primary Model: Normalized Output Throughput",
    subtitle_info=sub_v100_8b,
    paper=False
)

# fig = plot_speedup_all(v100_tp_8b, derived, 'Speedup Across TP Degree (V100)')

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
    paper=True,
)

# fig = plot_speedup_all(v100_tp_1b, derived, 'Speedup Across TP Degree (V100)')

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
    paper=True
)

# fig = plot_speedup_all(v100_tp_13b, derived, 'Speedup Across TP Degree (V100)')


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


#%%
for config, config_v in zip(a100_interconnect_tp4, v100_interconnect_tp4):
    config["linestyle"] = "--"
    config_v["linestyle"] = "--"
a100_interconnect = a100_interconnect_tp2 + a100_interconnect_tp4
v100_interconnect = v100_interconnect_tp2 + v100_interconnect_tp4


a100_interconnect_tp4_1b = build_interconnect_series('a100', 'llama-3.2-1b', tp=4)
a100_interconnect_tp2_1b = build_interconnect_series('a100', 'llama-3.2-1b', tp=2)


a100_interconnect_tp4_13b = build_interconnect_series('a100', 'llama-2-13b', tp=4, )
a100_interconnect_tp2_13b = build_interconnect_series('a100', 'llama-2-13b', tp=2)

for config_1b, config_13b in zip(a100_interconnect_tp4_1b, a100_interconnect_tp4_13b):
    config_1b["linestyle"] = "--"
    config_13b["linestyle"] = "--"

a100_interconnect_1b = a100_interconnect_tp2_1b + a100_interconnect_tp4_1b
a100_interconnect_13b = a100_interconnect_tp2_13b + a100_interconnect_tp4_13b

sub_a100_8b = dict(gpu='A100 GPUs', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')


fig_v100_thp_normalized = plot_normalized_throughput(
    v100_interconnect, data,
    title="Normalized Output Throughput",
    subtitle_info=sub_v100_8b,
    paper=False,
)

fig_a100_thp_normalized = plot_normalized_throughput(
    a100_interconnect, data,
    title="Normalized Output Throughput",
    subtitle_info=sub_a100_8b,
)

#%%
fig = plot_metric_vs_concurrency(a100_interconnect, data,
                                 metric="mean_itl_ms",
                                 ylabel="Mean ITL (ms)",
                                 title="ITL Across NCCL Transorts",
                                 paper=True)
fig = plot_metric_vs_concurrency(v100_interconnect, data,
                                 metric="mean_itl_ms",
                                 ylabel="Mean ITL (ms)",
                                 title="vITL Across NCCL Transorts",
                                 paper=True)
fig = plot_metric_vs_concurrency(a100_interconnect, data,
                                 metric="mean_ttft_ms",
                                 ylabel="Mean TTFT (ms)",
                                 title="TTFT Across NCCL Transports",
                                 log_scale=True,
                                 paper=True)
fig = plot_metric_vs_concurrency(v100_interconnect, data,
                                 metric="mean_ttft_ms",
                                 ylabel="Mean TTFT (ms)",
                                 title="vTTFT Across NCCL Transports",
                                 log_scale=True,
                                 paper=True)
#%%
a100_tp1_1b = a100_tp_1b[0]
a100_tp1_8b = a100_tp_8b[0]
a100_tp1_13b = a100_tp_13b[0]

# for i, s in enumerate(a100_interconnect):
#     print(f"{i} {s}")


idx = [1,2,3,4,5,6,7]
a100_subset = [a100_interconnect[i] for i in idx]

for config_1b, config_8b, config_13b in zip(a100_tp_1b_p2pdisabled, a100_tp_8b_p2pdisabled, a100_tp_13b_p2pdisabled):
    config_1b["linestyle"] = "--"
    config_8b["linestyle"] = "--"
    config_13b["linestyle"] = "--"
a100_1b_subset= [a100_tp1_1b] + a100_tp_1b + a100_tp_1b_p2pdisabled
a100_13b_subset= [a100_tp1_13b] + a100_tp_13b + a100_tp_13b_p2pdisabled
a100_8b_subset = [a100_tp1_8b] + a100_tp_8b + a100_tp_8b_p2pdisabled

fig = plot_throughput_efficiency(
    a100_interconnect,
    derived,
    title="Shy-fec: Throughput Efficiency Across Transports",
    subtitle_info=sub_a100_8b,
    paper=True
)
matplot2tikz.save("a100_efficiency_transport.tex")
fig = plot_throughput_efficiency(
    v100_interconnect,
    derived,
    title="funnotch: Throughput Efficiency Across Transports",
    subtitle_info=sub_v100_8b,
    paper=True
)
matplot2tikz.save("v100_efficiency_transport.tex")

fig = plot_normalized_throughput(a100_interconnect, data,
                                 title="a100 8b Normalized Throughput")

fig = plot_normalized_throughput(a100_interconnect_1b, data,
                                 title="a100 1b Normalized Throughput")
fig = plot_normalized_throughput(a100_interconnect_13b, data,
                                 title="a100 13b Normalized Throughput")


fig = plot_normalized_throughput(v100_interconnect, data,
                                 title="v100 Normalized Throughput")
fig = plot_normalized_throughput(a100_1b_subset, data,
                                 title="1b Normalized Throughput")
fig = plot_normalized_throughput(a100_8b_subset, data,
                                 title="8b Normalized Throughput")
fig = plot_normalized_throughput(a100_13b_subset, data,
                                 title="13b Normalized Throughput")

fig = plot_metric_vs_concurrency(a100_interconnect, derived,
                                 metric="throughput_efficiency",
                                 ylabel="Throughput Efficiency",
                                 title="a100")
fig = plot_metric_vs_concurrency(v100_interconnect, derived,
                                 metric="throughput_efficiency",
                                 ylabel="Throughput Efficiency",
                                 title="v100")


# %%
