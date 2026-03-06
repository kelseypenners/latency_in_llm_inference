#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from plot_utils import *

# load data
seqlen_df = pd.read_csv('../results/single-gpu/seqlen-sweep/averaged_seqlen_sweep_results.csv')
concurrency_df = pd.read_csv('../results/single-gpu/concurrency-sweep/averaged_concurrency_sweep_results.csv')

# gpu-specific
a100_seqlen_data = seqlen_df[seqlen_df['gpu_type'] == 'a100'].copy()
v100_seqlen_data = seqlen_df[seqlen_df['gpu_type'] == 'v100'].copy()

a100_concurrency_data = concurrency_df[concurrency_df['gpu_type'] == 'a100'].copy()
v100_concurrency_data = concurrency_df[concurrency_df['gpu_type'] == 'v100'].copy()

itldistributions_path = '../results/single-gpu/concurrency-sweep/itl_distributions.parquet'
# SEQLEN SWEEP PLOTS ----------------------------------------------------

def plot_input_output_heatmap(df, gpu_type, values, title, cmap='Oranges'):
    """ heatmap of metric across all ISL/OSL configs """
    
    pivot = df.pivot_table(values=values, 
                           index='random_output_len', columns='random_input_len')

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=ax, linecolor='white', linewidths=0.7, annot_kws={"size": 9})
    ax.invert_yaxis()
    ax.set_title(title, pad=20, fontweight='bold')
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='Concurrency = 1')
    ax.set_xlabel('Input Sequence Length (tokens)', labelpad=10)
    ax.set_ylabel('Output Sequence Length (tokens)', labelpad=10)
    plt.tight_layout()

    return fig

def plot_heatmap_comparison(a100_df, v100_df, values, title, cmap='Oranges', same_range=False):
    """ comparison of heatmaps between GPUs for same metric """
    dfs = [a100_df, v100_df]
    gpu_types = ['A100 40GB', 'V100 32GB']
    pivots = [df.pivot_table(values=values, index='random_output_len', columns='random_input_len')
              for df in dfs]
    if same_range:
        vmin = min(p.min().min() for p in pivots)
        vmax = max(p.max().max() for p in pivots)
    else: 
        vmin = None 
        vmax = None

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, pivot, gpu_type in zip(axes, pivots, gpu_types):
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=ax, linecolor='white', 
                    linewidths=0.7, annot_kws={"size": 9}, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(gpu_type, pad=15, fontweight='bold')
        ax.set_xlabel('Input Sequence Length (tokens)', labelpad=10)
        ax.set_ylabel('Output Sequence Length (tokens)', labelpad=10)
    fig.suptitle(title, fontweight='bold')
    fig.text(0.5, 0.93, 'Llama-3.1-8B  |  Concurrency = 1',
             ha='center', va='top', fontsize=10, color='grey')
    plt.tight_layout()

    return fig

def plot_ttft_vs_isl(df, gpu_type, osl=None, title=None):
    """ TTFT vs ISL, one line per OSL value (or a fixed OSL) """

    if osl == None:
        osl_values = sorted(df['random_output_len'].unique())
    else:
        osl_values = [osl]
    title = title or 'TTFT Scales Linearly with ISL, Independent of OSL'

    fig, ax = plt.subplots(figsize=(7,5))
    for osl in osl_values:
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'], marker='o', label=f"OSL={osl}", linewidth=2)
    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='Concurrency = 1')
    ax.set_xlabel('Input Sequence Length (ISL) - (tokens)')
    ax.legend(title='Output Length', loc='upper left')
    ax.set_ylabel('TTFT (ms)')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    return fig

def plot_ttft_vs_isl_comparison(a100_df, v100_df, osl=512, title=None):
    """ TTFT vs ISL comparison between GPUs"""

    title = title or f'TTFT vs ISL - A100 vs V100 (OSL={osl})'

    labels = ['A100 40GB', 'V100 32GB']
    dfs = [a100_df, v100_df]
    fig, ax = plt.subplots(figsize=(7,5))
    for i in range(2):
        df = dfs[i]
        label = labels[i]
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'], marker='o', label=label, linewidth=2)
    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type='A100 40GB vs V100 32GB', model='Llama-3.1-8B', extra='Concurrency = 1')
    ax.set_xlabel('Input Sequence Length (ISL) - (tokens)')
    ax.legend(title='Output Length', loc='upper left')
    ax.set_ylabel('TTFT (ms)')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return fig

def plot_ttft_vs_osl_per_isl(df):
    """ multiple plots: TTFT vs OSL for each ISL """

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
    """ multiple plots: TTFT vs ISL for each OSL """

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

def plot_e2el_vs_osl(df, gpu_type, isl=None, title=None):
    """ E2E latency vs OSL, one line per ISL (or fixed ISL) """
    
    if isl == None:
        isl_values = sorted(df['random_input_len'].unique())
    else:
        isl_values = [isl]
    title = title or 'Impact of Output Sequence Length (OSL) on E2E Latency'
    
    fig, ax = plt.subplots(figsize=(7,5))
    for isl in isl_values:
        data = df[df['random_input_len'] == isl]
        ax.plot(data['random_output_len'], data['mean_e2el_ms'], marker='o', label=f"ISL={isl}", linewidth=2)
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlabel('OSL (tokens)')
    ax.legend(title='Input Length')
    ax.set_ylabel('E2E Latency (ms)')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return fig

def plot_e2el_vs_isl_per_osl(df):
    """ multiple plots: E2E latency vs ISL for each OSL """

    osl_values = sorted(df['random_output_len'].unique())
    title = title or 'E2E Latency vs ISL (fixed OSL)'

    fig, axes = plt.subplots(1, len(osl_values), figsize=(4*len(osl_values), 4), sharey=True)
    for ax, osl in zip(axes, osl_values):
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_e2el_ms'], marker='o')
        ax.set_title(f'OSL={osl}')
        ax.set_xlabel('ISL (tokens)')
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel('E2E Latency (ms)')
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_metrics_vs_concurrency(df, gpu_type, metric, label, title):
    """ metric versus max concurrency """

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(df['max_concurrency'], df[metric], marker='o')
    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency (max in-flight requests)')
    ax.set_ylabel(label)
    ax.set_xscale('log', base=2)
    ax.set_xticks(df['max_concurrency'])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    return fig

def plot_mean_vs_p99(df, gpu_type, metric="itl", title=None, p90=True):
    """ mean vs P99 for ITL vs concurrency """
    title = title or f'{metric.upper()}: Mean vs Tail Latency vs Concurrency'

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df['max_concurrency'], df[f'mean_{metric}_ms'], marker='o', linewidth=2, label='Mean')
    ax.plot(df['max_concurrency'], df[f'p99_{metric}_ms'], marker='s', linewidth=2, linestyle='--', label='P99 (slowest 1% of tokens)')
    ax.plot(df['max_concurrency'], df[f'p90_{metric}_ms'], marker='s', linewidth=2, linestyle='--', label='P90 (slowest 10% of tokens)')
    ax.fill_between(df['max_concurrency'], df[f'mean_{metric}_ms'], df[f'p99_{metric}_ms'], alpha=0.12, label='Mean / P99 Gap')

    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency', labelpad=8)
    ax.set_ylabel(f'{metric.upper()} (ms)', labelpad=8)
    ax.set_xticks(df['max_concurrency'])
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    return fig

def plot_run_variance(seqlen_df, concurrency_df, gpu_type, stds=1, osl=512):
    """ TTFT run variance (seqlen) and throughput run variance (concurrency) """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # TTFT variance
    ax = axes[0]
    fixed_osl = seqlen_df[seqlen_df['random_output_len'] == osl]
    ax.plot(fixed_osl['random_input_len'], fixed_osl['mean_ttft_ms'], label= 'Mean TTFT')
    if 'std_runs_mean_ttft_ms' in fixed_osl.columns:
        ax.fill_between(fixed_osl['random_input_len'], fixed_osl['mean_ttft_ms'] - stds * fixed_osl['std_runs_mean_ttft_ms'], fixed_osl['mean_ttft_ms'] + stds * fixed_osl['std_runs_mean_ttft_ms'], alpha=0.2, label=f'+/-{stds} std across runs')

    ax.set_xlabel('Input Sequence Length (tokens)', labelpad=8)
    ax.set_ylabel('TTFT (ms)', labelpad=8)
    ax.set_title('TTFT: Run-to-Run Variance', fontweight='bold', pad=15)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra=f'OSL={osl}, Concurrency=1')
    ax.set_xticks(sorted(fixed_osl['random_input_len'].unique()))
    ax.grid(True, axis='y', alpha=0.4)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(concurrency_df['max_concurrency'], concurrency_df['output_throughput'], label='Mean Throughput')

    if 'std_runs_output_throughput' in concurrency_df.columns:
        ax.fill_between(concurrency_df['max_concurrency'], concurrency_df['output_throughput'] - stds * concurrency_df['std_runs_output_throughput'], concurrency_df['output_throughput'] + stds * concurrency_df['std_runs_output_throughput'], alpha=0.2, label=f'+/-{stds} std across runs')

    ax.set_xlabel('Concurrency (max in-flight requests)', labelpad=8)
    ax.set_ylabel('Output Throughput (tokens/s)', labelpad=8)
    ax.set_title('Throughput: Run-to-Run Variance', fontweight='bold', pad=15)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='ISL/OSL = 512/512')
    ax.set_xticks(concurrency_df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.4)
    ax.legend(fontsize=9)
    
    return fig

def plot_itl_boxplot(parquet_path, gpu_type, title=None):
    """ per-token ITL distribution as box-whisker plot across concurrency levels """
    title = title or 'Per-Token ITL Distribution vs Concurrency'

    df = pd.read_parquet(parquet_path)
    if gpu_type.lower().replace(' ', '') in ['a10040gb', 'a100']:
        df = df[df['gpu_type'] == 'a100']
    else:
        df = df[df['gpu_type'] == 'v100']

    df_exploded = df.explode('itls_ms')
    df_exploded['itls_ms'] = df_exploded['itls_ms'].astype(float)

    concurrency_levels = sorted(df_exploded['max_concurrency'].unique())
    data = [df_exploded[df_exploded['max_concurrency'] == c]['itls_ms'].values for c in concurrency_levels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp =ax.boxplot(data, labels=concurrency_levels, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#4C9BE8')
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('#E85C4C')
        median.set_linewidth(2)
    for whisker in bp['whiskers']:
        whisker.set_color('#333333')
        whisker.set_linewidth(1.2)
    for cap in bp['caps']:
        cap.set_color('#333333')
        cap.set_linewidth(1.2)
    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra=f'ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency (max in-flight requests)', labelpad=8)
    ax.set_ylabel('ITL (ms)', labelpad=8)
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    return fig

# A100 PLOTS ----------------------------------------------------
#%% [markdown] 
# ## A100 Sequence Length Sweep 
data = a100_seqlen_data
gpu_type = 'A100 40GB'
fig = plot_input_output_heatmap(df = data,
                                gpu_type = gpu_type,
                                values ='mean_ttft_ms',
                                title ='Time to First Token (TTFT) (ms)',
                                cmap = 'coolwarm')
fig = plot_input_output_heatmap(df = data,
                                gpu_type = gpu_type,
                                values = 'output_throughput',
                                title ='Output Throughput (tokens/s)',
                                cmap = 'coolwarm_r')
fig = plot_input_output_heatmap(df = data,
                                gpu_type = gpu_type,
                                values = 'total_token_throughput',
                                title ='Total Token Throughput (tokens/s)',
                                cmap = 'coolwarm_r')

fig = plot_ttft_vs_isl(df = data, gpu_type = gpu_type)

fig = plot_e2el_vs_osl(df = data, gpu_type = gpu_type)

#%% [markdown] 
# ## A100 Concurrency Sweep 
data = a100_concurrency_data
fig = plot_metrics_vs_concurrency(df = data,
                                  gpu_type = gpu_type,
                                  metric = 'output_throughput',
                                  label = 'Output Throughput (tokens/s)',
                                  title = 'Output Throughput vs Concurrency')
fig = plot_metrics_vs_concurrency(df = data,
                                  metric ='mean_itl_ms', 
                                  label ='ITL (ms)',
                                  gpu_type = gpu_type,
                                  title = 'ITL (ms) vs Concurrency')
fig = plot_metrics_vs_concurrency(df = data,
                                  gpu_type = gpu_type,
                                  metric ='mean_ttft_ms', 
                                  label = 'TTFT (ms)',
                                  title = 'TTFT (ms) vs Concurrency')
fig = plot_metrics_vs_concurrency(df = data,
                                  gpu_type = gpu_type,
                                  metric ='mean_tpot_ms', 
                                  label =' TPOT (ms)',
                                  title = 'TPOT (ms) vs Concurrency')
fig = plot_mean_vs_p99(df = data,
                       gpu_type = gpu_type,
                       metric ='itl')
fig = plot_mean_vs_p99(df = data,
                       gpu_type = gpu_type,
                       metric ='tpot')


# V100 PLOTS ----------------------------------------------------
#%% [markdown] 
# ## V100 Sequence Length Sweep 
data = v100_seqlen_data
gpu_type = 'V100 32GB'
fig = plot_input_output_heatmap(df = data,
                                gpu_type = gpu_type,
                                values ='mean_ttft_ms',
                                title ='Time to First Token (TTFT) (ms)',
                                cmap = 'coolwarm')
fig = plot_input_output_heatmap(df = data,
                                gpu_type = gpu_type,
                                values = 'output_throughput',
                                title ='Output Throughput (tokens/s)',
                                cmap = 'coolwarm_r')
fig = plot_input_output_heatmap(df = data,
                                gpu_type = gpu_type,
                                values = 'total_token_throughput',
                                title ='Total Token Throughput (tokens/s)',
                                cmap = 'coolwarm_r')

fig = plot_ttft_vs_isl(df = data, gpu_type = gpu_type)

fig = plot_e2el_vs_osl(df = data, gpu_type = gpu_type)

# COMPARISON ----------------------------------------------------
#%% [markdown]
# ## A100 vs V100 Comparison
fig = plot_ttft_vs_isl_comparison(a100_df = a100_seqlen_data,
                                  v100_df = v100_seqlen_data,
                                  osl = 512)

fig = plot_heatmap_comparison(a100_df = a100_seqlen_data,
                              v100_df = v100_seqlen_data,
                              values = 'mean_ttft_ms',
                              title = 'Time to First Token (TTFT) (ms)',
                              cmap = 'coolwarm')
fig = plot_heatmap_comparison(a100_df = a100_seqlen_data,
                              v100_df = v100_seqlen_data,
                              values = 'output_throughput',
                              title = 'Output Throughput (tokens/s)',
                              cmap = 'coolwarm_r')

# RUN VARIANCE ----------------------------------------------------
#%% [markdown]
# ## Run Variance
seqlen_data = a100_seqlen_data
concurrency_data = a100_concurrency_data
gpu_type = 'A100 40GB'
fig = plot_run_variance(seqlen_data, concurrency_data, gpu_type=gpu_type, stds=1, osl=512)

seqlen_data = v100_seqlen_data
concurrency_data = a100_concurrency_data
gpu_type = 'V100 32GB'
fig = plot_run_variance(seqlen_data, concurrency_data, gpu_type=gpu_type, stds=1, osl=512)

plot_itl_boxplot(parquet_path = itldistributions_path, 
                 gpu_type = 'A100 40GB')

# %%
