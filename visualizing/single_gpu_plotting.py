#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_utils import *

# load data
sweep_df = pd.read_csv('../results/single-gpu/23-02-2026/averaged_sweep_results.csv')
batch_sweep_df = pd.read_csv('../results/single-gpu/25-02-2026/averaged_concurrency_sweep_results.csv')

batch_sweep_df = pd.read_csv('../results/single-gpu/27-02-2026/averaged_batch_sweep_results.csv')

# gpu-specific
a100_data = sweep_df[sweep_df['gpu_type'] == 'a100'].copy()
v100_data = sweep_df[sweep_df['gpu_type'] == 'v100'].copy()

def plot_input_output_heatmap(df, values, title):
    """ plot heatmap for input and output lengths and given value """
    
    pivot = df.pivot_table(values=values, 
                           index='random_output_len', columns='random_input_len')

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Oranges', ax=ax, linecolor='white', linewidths=0.7, annot_kws={"size": 9})
    ax.invert_yaxis()
    ax.set_title(title, pad=20, fontweight='bold')
    ax.text(0.5, 1.03, 'A100 40GB  |  Llama-3.1-8B  |  Concurrency = 1',
        transform=ax.transAxes, ha='center', fontsize=8, color='gray')
    ax.set_xlabel('Input Sequence Length (ISL)', labelpad=10)
    ax.set_ylabel('Output Sequence Length (OSL)', labelpad=10)
    plt.tight_layout()

    return fig

def plot_ttft_vs_isl(df, osl=None):
    """ plot impact of ISL on TTFT """

    if osl == None:
        osl_values = sorted(df['random_output_len'].unique())
    else:
        osl_values = [osl]
    fig, ax = plt.subplots(figsize=(7,5))

    for osl in osl_values:
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'], marker='o', label=f"OSL={osl}", linewidth=2)
    ax.set_title(f'TTFT Scales Linearly with ISL, Independent of OSL', fontweight='bold', pad=20)
    ax.text(0.5, 1.03, 'A100 40GB  |  Llama-3.1-8B  |  Concurrency = 1',
        transform=ax.transAxes, ha='center', fontsize=8, color='gray')
    ax.set_xlabel('Input Sequence Length (ISL) - (tokens)')
    ax.legend(title='Output Length', loc='upper left')
    ax.set_ylabel('TTFT (ms)')
    ax.grid(True, alpha=0.4)
    #ax.set_xticks(sorted(df['random_input_len'].unique()))

    plt.tight_layout()
    return fig

def plot_ttft_vs_osl_per_isl(df):
    """ plot impact of OSL on TTFT for each ISL config """

    isl_values = sorted(df['random_input_len'].unique())
    fig, axes = plt.subplots(1, len(isl_values), figsize=(4*len(isl_values), 4), sharey=True)

    for ax, isl in zip(axes, isl_values):
        data = df[df['random_input_len'] == isl]
        ax.plot(data['random_output_len'], data['mean_ttft_ms'], marker='o')
        ax.set_title(f'ISL={isl}')
        ax.set_xlabel('OSL (tokens)')
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel('TTFT (ms)')
    plt.suptitle('TTFT vs OSL (fixed ISL)', fontweight='bold')

    plt.tight_layout()
    return fig

def plot_ttft_vs_isl_per_osl(df):
    """ plot impact of ISL on TTFT for each OSL config """

    osl_values = sorted(df['random_output_len'].unique())
    fig, axes = plt.subplots(1, len(osl_values), figsize=(4*len(osl_values), 4), sharey=True)

    for ax, osl in zip(axes, osl_values):
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'], marker='o')
        ax.set_title(f'OSL={osl}')
        ax.set_xlabel('ISL (tokens)')
        ax.grid(True, alpha=0.4)
    
    axes[0].set_ylabel('TTFT (ms)')
    plt.suptitle('TTFT vs ISL (fixed OSL)', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_e2el_vs_osl(df, isl=None):
    """ plot impact of OSL on E2E latency """
    
    if isl == None:
        isl_values = sorted(df['random_input_len'].unique())
    else:
        isl_values = [isl]

    fig, ax = plt.subplots(figsize=(7,5))

    for isl in isl_values:
        data = df[df['random_input_len'] == isl]
        ax.plot(data['random_output_len'], data['mean_e2el_ms'], marker='o', label=f"ISL={isl}", linewidth=2)
    ax.set_title(f'Impact of Output Sequence Length (OSL) on E2E Latency', fontweight='bold')
    ax.set_xlabel('OSL (tokens)')
    ax.legend(title='Input Length')
    ax.set_ylabel('E2E Latency (ms)')
    ax.grid(True, alpha=0.4)
    #ax.set_xticks(sorted(df['random_output_len'].unique()))

    plt.tight_layout()
    return fig



def plot_e2el_vs_isl_per_osl(df):
    """ plot impact of ISL on TTFT for each OSL config """

    osl_values = sorted(df['random_output_len'].unique())
    fig, axes = plt.subplots(1, len(osl_values), figsize=(4*len(osl_values), 4), sharey=True)

    for ax, osl in zip(axes, osl_values):
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_e2el_ms'], marker='o')
        ax.set_title(f'OSL={osl}')
        ax.set_xlabel('ISL (tokens)')
        ax.grid(True, alpha=0.4)
    
    axes[0].set_ylabel('E2E Latency')
    plt.suptitle('E2E Latency vs ISL (fixed OSL)', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_metrics_vs_batch(df, metric, label, title):
    """ plot metrics versus max concurrency """
    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(df['max_concurrency'], df[metric], marker='o')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.text(0.5, 1.03, 'A100 40GB  |  Llama-3.1-8B  |  ISL / OSL = 512 / 512',
        transform=ax.transAxes, ha='center', fontsize=8, color='gray')
    ax.set_xlabel('Concurrency (max in-flight requests)')
    ax.set_ylabel(label)
    ax.set_xscale('log', base=2)
    ax.set_xticks(df['max_concurrency'])
    ax.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    return fig

def plot_mean_vs_p99(df, metric="itl", title="ITL Tail Latency Inflation with Concurrency", p90=True):
    """ mean vs P99 for ITL vs concurrency """
    fig, ax = plt.subplots(figsize=(6, 5))

    
    ax.plot(df['max_concurrency'], df[f'mean_{metric}_ms'], marker='o', linewidth=2, label='Mean')
    ax.plot(df['max_concurrency'], df[f'p99_{metric}_ms'], marker='s', linewidth=2, linestyle='--', label='P99')
    ax.plot(df['max_concurrency'], df[f'p90_{metric}_ms'], marker='s', linewidth=2, linestyle='--', label='P90')
    ax.plot(df['max_concurrency'], df[f'p75_{metric}_ms'], marker='s', linewidth=2, linestyle='--', label='P75')
    ax.fill_between(df['max_concurrency'], df[f'mean_{metric}_ms'], df[f'p99_{metric}_ms'],
                    alpha=0.12, label='Mean / P99 Gap')

    ax.set_title(f'{metric.upper()}: Mean vs P99')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.text(0.5, 1.03, 'A100 40GB  |  Llama-3.1-8B  |  ISL / OSL = 512 / 512',
        transform=ax.transAxes, ha='center', fontsize=8, color='gray')
    ax.set_xlabel('Concurrency', labelpad=8)
    ax.set_ylabel(f'{metric.upper()} (ms)', labelpad=8)
    ax.set_xticks(df['max_concurrency'])
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    return fig

def plot_seqlen_4panel(df, std_across_runs=False, stds=1):
    """4-panel summary of sequence length sweep: TTFT, ITL, throughput, E2E latency."""
    
    if std_across_runs:
        metrics = [
            ('random_input_len', 'mean_ttft_ms',      'std_runs_mean_ttft_ms',  'TTFT (ms)'),
            ('random_output_len', 'mean_itl_ms',       'std_runs_mean_itl_ms',   'ITL (ms)'),
            ('random_input_len', 'output_throughput',  'std_runs_output_throughput',   'Output Throughput (tokens/s)'),
            ('random_output_len', 'mean_e2el_ms',      'std_runs_mean_e2el_ms',  'E2E Latency (ms)'),
        ]
    else:
        metrics = [
            ('random_input_len', 'mean_ttft_ms',      'std_ttft_ms',  'TTFT (ms)'),
            ('random_output_len', 'mean_itl_ms',       'std_itl_ms',   'ITL (ms)'),
            ('random_input_len', 'output_throughput',  None,   'Output Throughput (tokens/s)'),
            ('random_output_len', 'mean_e2el_ms',      'std_e2el_ms',  'E2E Latency (ms)'),
        ]

    by_isl = df[df['random_input_len'] == 512]
    by_osl = df[df['random_output_len'] == 512]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (xlabel, col, std_col, ylabel) in zip(axes, metrics):
        if xlabel == 'random_input_len':
            df = by_osl
        else:
            df = by_isl
        ax.plot(df[xlabel], df[col],
                marker='o', linewidth=2.5, markersize=6)

        if std_col and std_col in df.columns:
            ax.fill_between(df[xlabel],
                            df[col] - stds*df[std_col],
                            df[col] + stds*df[std_col],
                            alpha=0.15, label=f'±{stds} std (within run)')

        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_title(ylabel)
        ax.set_xticks(df[xlabel])
        ax.grid(True, axis='y', alpha=0.4)

    fig.suptitle('Sequence Length Sweep – A100 40GB  |  Llama-3.1-8B  |  Concurrency = 1',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    return fig

def plot_concurrency_4panel(df, std_across_runs=False, stds=1):
    """4-panel summary of concurrency sweep: TTFT, ITL, throughput, E2E latency."""
    
    if std_across_runs:
        metrics = [
            ('mean_ttft_ms',      'std_runs_mean_ttft_ms',  'TTFT (ms)'),
            ('mean_itl_ms',       'std_runs_mean_itl_ms',   'ITL (ms)'),
            ('output_throughput',  'std_runs_output_throughput',   'Output Throughput (tokens/s)'),
            ('mean_e2el_ms',      'std_runs_mean_e2el_ms',  'E2E Latency (ms)'),
        ]
    else:
        metrics = [
            ('mean_ttft_ms',      'std_ttft_ms',  'TTFT (ms)'),
            ('mean_itl_ms',       'std_itl_ms',   'ITL (ms)'),
            ('output_throughput',  None,           'Output Throughput (tokens/s)'),
            ('mean_e2el_ms',      'std_e2el_ms',  'E2E Latency (ms)'),
        ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (col, std_col, ylabel) in zip(axes, metrics):
        ax.plot(df['max_concurrency'], df[col],
                marker='o', linewidth=2.5, markersize=6)

        if std_col and std_col in df.columns:
            ax.fill_between(df['max_concurrency'],
                            df[col] - stds*df[std_col],
                            df[col] + stds*df[std_col],
                            alpha=0.15, label=f'±{stds} std (within run)')

        ax.set_xlabel('Concurrency', labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_title(ylabel)
        ax.set_xscale('log', base=2)
        ax.set_xticks(df['max_concurrency'])
        ax.grid(True, axis='y', alpha=0.4)

    fig.suptitle('Concurrency Sweep – A100 40GB  |  Llama-3.1-8B  |  ISL=512, OSL=512',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    return fig

# %% TTFT VISUALS

# TTFT vs ISL (across all OSLs)
fig = plot_ttft_vs_isl(a100_data)

# to look at trends for ISL and OSL 
#fig = plot_ttft_vs_isl_per_osl(a100_data)
#fig = plot_ttft_vs_osl_per_isl(a100_data)

# %% HEATMAPS

# heatmaps for all metrics
fig = plot_input_output_heatmap(a100_data, values='output_throughput', title='Output Throughput (tokens/s)')
fig = plot_input_output_heatmap(a100_data, values='total_token_throughput', title='Total Token Throughput (tokens/s)')
fig = plot_input_output_heatmap(a100_data, values='mean_e2el_ms', title='Average E2E Latency (ms)')
#fig = plot_input_output_heatmap(a100_data, values='mean_itl_ms', title='ITL (ms)')
#fig = plot_input_output_heatmap(a100_data, values='mean_tpot_ms', title='TPOT (ms)')
fig = plot_input_output_heatmap(a100_data, values='mean_ttft_ms', title='TTFT (ms)')

# %% E2E LATENCY VISUALS

# E2EL vs ISL and OSL
fig = plot_e2el_vs_osl(a100_data)
#fig = plot_e2el_vs_isl_per_osl(a100_data)

# %% BATCH SIZE VISUALS

# batch size versus metrics
# fig = plot_metrics_vs_batch(batch_sweep_df, "mean_ttft_ms", "TTFT (ms)", "TTFT vs Concurrency")
# fig = plot_metrics_vs_batch(batch_sweep_df, "mean_itl_ms", "ITL (ms)", "ITL vs Concurrency")
# fig = plot_metrics_vs_batch(batch_sweep_df, "mean_tpot_ms", "TPOT (ms)", "TPOT vs Concurrency")
# fig = plot_metrics_vs_batch(batch_sweep_df, "mean_e2el_ms", "E2EL (ms)", "E2EL vs Concurrency")
fig = plot_metrics_vs_batch(batch_sweep_df, "output_throughput", "Output Throughput (tokens/s)", "Output Throughput Scales Nonlinearly with Concurrency")  
#fig = plot_metrics_vs_batch(batch_sweep_df, "total_token_throughput", "Total Token Throughput", "Total Token Throughput vs Concurrency")  

fig = plot_concurrency_4panel(batch_sweep_df)
fig = plot_concurrency_4panel(batch_sweep_df, std_across_runs=True, stds=1)

fig = plot_seqlen_4panel(a100_data, std_across_runs=True, stds=20)

# %%
fig = plot_mean_vs_p99(batch_sweep_df)
fig = plot_mean_vs_p99(batch_sweep_df, 'ttft')
fig = plot_mean_vs_p99(batch_sweep_df, 'tpot', title="TPOT vs Concurrency")


# the ones alex liked
fig = plot_ttft_vs_isl(a100_data)
fig = plot_input_output_heatmap(a100_data, values='output_throughput', title='Output Throughput (tokens/s)')
fig = plot_metrics_vs_batch(batch_sweep_df, "output_throughput", "Output Throughput (tokens/s)", "Output Throughput Scales Nonlinearly with Concurrency")  

fig = plot_mean_vs_p99(batch_sweep_df)