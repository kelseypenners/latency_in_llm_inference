#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import ast

from plot_utils import *

# load sequence length sweep data
seqlen_df = pd.read_csv('../results/single-gpu/seqlen-sweep/averaged_seqlen_sweep_results.csv')
# load baseline concurrency sweep data
concurrency_df = pd.read_csv('../results/single-gpu/concurrency-sweep/baseline/averaged_concurrency_sweep_baseline_results.csv')
# load fine-grained concurrency sweep data
concurrency_df = pd.read_csv('../results/single-gpu/concurrency-sweep/fine-grained/averaged_concurrency_sweep_fine_grained_results.csv')
# load itl data
itldistributions_path = '../results/single-gpu/concurrency-sweep/fine-grained/itl_distributions.parquet'
# load prompt type data
prompttype_df = pd.read_csv('../results/single-gpu/dataset-comparison/prompttype_10runs_1prompt.csv')
# load prompt type data
prompttype_df_2 = pd.read_csv('../results/single-gpu/dataset-comparison/prompttype_2runs_100prompts.csv')

# gpu-specific
a100_seqlen_data = seqlen_df[seqlen_df['gpu_type'] == 'a100'].copy()
v100_seqlen_data = seqlen_df[seqlen_df['gpu_type'] == 'v100'].copy()

a100_concurrency_data = concurrency_df[concurrency_df['gpu_type'] == 'a100'].copy()
v100_concurrency_data = concurrency_df[concurrency_df['gpu_type'] == 'v100'].copy()


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

# CONCURRENCY SWEEP PLOTS ----------------------------------------------------
def plot_metrics_vs_concurrency(df, gpu_type, metric, label, title):
    """ metric versus max concurrency """

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(df['max_concurrency'], df[metric], marker='o')
    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency (max in-flight requests)')
    ax.set_ylabel(label)
    ax.set_xticks(df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    return fig

def plot_mean_vs_p99(df, gpu_type, metric="itl", title=None, p90=True):
    """ mean vs P99 for ITL vs concurrency """
    title = title or f'{metric.upper()}: Mean vs Tail Latency vs Concurrency'

    fig, ax = plt.subplots(figsize=(7, 5))
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

def plot_itl_boxplot(parquet_path, gpu_type, title=None, showfliers=False):
    """ ITL distribution box-whisker plot across concurrency levels """

    title = title or 'ITL Distribution vs Concurrency'

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
    bp =ax.boxplot(data, labels=concurrency_levels, showfliers=showfliers, patch_artist=True)
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
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    return fig

def plot_itl_boxplot_outliers(parquet_path, gpu_type, title=None):
    """ ITL distribution box-whisker plot across concurrency levels with broken y-axis to show. outliers"""

    title = title or 'ITL Distribution vs Concurrency'
    df = pd.read_parquet(parquet_path)
    if gpu_type.lower().replace(' ', '') in ['a10040gb', 'a100']:
        df = df[df['gpu_type'] == 'a100']
    else:
        df = df[df['gpu_type'] == 'v100']

    df_exploded = df.explode('itls_ms')
    df_exploded['itls_ms'] = df_exploded['itls_ms'].astype(float)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6),
        gridspec_kw={'height_ratios': [2, 2]})

    concurrency_levels = sorted(df_exploded['max_concurrency'].unique())
    data = [df_exploded[df_exploded['max_concurrency'] == c]['itls_ms'].values for c in concurrency_levels]

    # bottom plot (normal range)
    ax2.boxplot(data, labels=concurrency_levels, showfliers=False)
    ax2.set_ylim(0, 60)

    # top plot (outliers)
    ax1.boxplot(data, labels=concurrency_levels, showfliers=True)
    ax1.set_ylim(100, max([max(d) for d in data]))

    # styling break
    ax1.grid(True, axis='y', alpha=0.4)
    ax2.grid(True, axis='y', alpha=0.4)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()
    ax1.set_ylabel('ITL (ms)')
    ax2.set_xlabel('Concurrency')
    ax2.tick_params(axis='x', rotation=45)

# COMPARISON PLOTS ----------------------------------------------------
def plot_run_variance(seqlen_df, concurrency_df, gpu_type, stds=1, osl=512):
    """ TTFT run variance (seqlen) and throughput run variance (concurrency) """
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

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

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, pivot, gpu_type in zip(axes, pivots, gpu_types):
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=ax, linecolor='white', 
                    linewidths=0.7, annot_kws={"size": 9}, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(gpu_type, pad=15, fontweight='bold')
        ax.set_xlabel('Input Sequence Length (tokens)', labelpad=10)
        ax.set_ylabel('Output Sequence Length (tokens)', labelpad=10)
    fig.suptitle(title, fontweight='bold')
    fig.text(0.5, 0.945, 'Llama-3.1-8B  |  Concurrency = 1',
             ha='center', va='top', fontsize=10, color='grey')
    plt.tight_layout()

    return fig
def plot_ttft_vs_isl_comparison(a100_df, v100_df, osl=512, title=None):
    """ TTFT vs ISL comparison between GPUs"""

    title = title or f'TTFT vs ISL - A100 vs V100)'

    labels = ['A100 40GB', 'V100 32GB']
    dfs = [a100_df, v100_df]
    fig, ax = plt.subplots(figsize=(7,5))
    for i in range(2):
        df = dfs[i]
        label = labels[i]
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'], marker='o', label=label, linewidth=2)
    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type='A100 40GB vs V100 32GB', model='Llama-3.1-8B', extra=f'Concurrency = 1  | OSL={osl}')
    ax.set_xlabel('Input Sequence Length (ISL) - (tokens)')
    ax.legend(title='GPU Type', loc='upper left')
    ax.set_ylabel('TTFT (ms)')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return fig

def plot_metrics_vs_concurrency_comparison(a100_df, v100_df, metric, ylabel, title):
    """ metric versus max concurrency comparison between GPUs """

    labels = ['A100 40GB', 'V100 32GB']
    dfs = [a100_df, v100_df]
    fig, ax = plt.subplots(figsize=(7,5))
    for i in range(2):
        df =dfs[i]
        label = labels[i]
        ax.plot(df['max_concurrency'], df[metric], marker='o', label=label, linewidth=2)
    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, 'A100 vs V100', model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency (max in-flight requests)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(title='GPU Type', loc='upper left')
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    return fig

def plot_mean_vs_p99_comparison(a100_df, v100_df, metric='itl', title=None):
    """ mean vs P99 comparison between GPUs """

    title = title or f'{metric.upper()}: Mean vs Tail Latency - A100 vs V100'
    
    dfs = [a100_df, v100_df]
    gpu_types = ['A100 40GB', 'V100 32GB']

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, df, gpu_type in zip(axes, dfs, gpu_types):
        ax.plot(df['max_concurrency'], df[f'mean_{metric}_ms'], marker='o', linewidth=2, label='Mean')
        ax.plot(df['max_concurrency'], df[f'p99_{metric}_ms'], marker='s', linewidth=2, linestyle='--', label='P99 (slowest 1%)')
        ax.plot(df['max_concurrency'], df[f'p90_{metric}_ms'], marker='s', linewidth=2, linestyle='--', label='P90 (slowest 10%)')
        ax.fill_between(df['max_concurrency'], df[f'mean_{metric}_ms'], df[f'p99_{metric}_ms'], alpha=0.12, label='Mean / P99 Gap')
        ax.set_title(gpu_type, fontweight='bold', pad=15)
        ax.set_xlabel('Concurrency', labelpad=8)
        ax.set_ylabel(f'{metric.upper()} (ms)', labelpad=8)
        ax.set_xticks(df['max_concurrency'])
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, axis='y', alpha=0.4)
        ax.legend(fontsize=9)
    fig.suptitle(title, fontweight='bold')
    fig.text(0.5, 0.93, 'Llama-3.1-8B  |  ISL / OSL = 512 / 512',
             ha='center', va='top', fontsize=10, color='grey')
    plt.tight_layout()
    return fig

def plot_itl_boxplot_comparison(parquet_path, title=None):
    """ ITL distribution box-whisker comparison between A100 and V100 """

    title = title or 'ITL Distribution vs Concurrency - A100 vs V100'

    df_all = pd.read_parquet(parquet_path)
    gpu_types = ['a100', 'v100']
    gpu_labels = ['A100 40GB', 'V100 32GB']

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, gpu, label in zip(axes, gpu_types, gpu_labels):
        df = df_all[df_all['gpu_type'] == gpu]
        df_exploded = df.explode('itls_ms')
        df_exploded['itls_ms'] = df_exploded['itls_ms'].astype(float)

        concurrency_levels = sorted(df_exploded['max_concurrency'].unique())
        data = [df_exploded[df_exploded['max_concurrency'] == c]['itls_ms'].values for c in concurrency_levels]

        bp = ax.boxplot(data, labels=concurrency_levels, showfliers=False, patch_artist=True)
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
        ax.set_title(label, fontweight='bold', pad=15)
        ax.set_xlabel('Concurrency (max in-flight requests)', labelpad=8)
        ax.set_ylabel('ITL (ms)', labelpad=8)
        ax.grid(True, axis='y', alpha=0.4)
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(title, fontweight='bold')
    fig.text(0.5, 0.93, 'Llama-3.1-8B  |  ISL / OSL = 512 / 512',
             ha='center', va='top', fontsize=10, color='grey')
    plt.tight_layout()
    return fig

# DATASET COMPARISON PLOT ----------------------------------------------------
def plot_prompt_type_ttft(df, gpu_type, isl=512, osl=64, title=None):
    title = title or 'TTFT by Prompt Type'

    def get_prompt_type(row):
        path = str(row.get('dataset-path', '') or '')
        if 'easy' in path:
            return 'easy'
        elif 'hard' in path:
            return 'hard'
        elif str(row.get('dataset-name', '')).lower() == 'random':
            return 'random'
        return None
    df['prompt_type'] = df.apply(get_prompt_type, axis=1)
    df = df[df['prompt_type'].notna()]

    keys   = ['easy', 'hard', 'random']
    labels = ['Easy\n(repetitive)', 'Hard\n(coherent text)', 'Random\n(vLLM random)']
    colors = ['cornflowerblue', 'firebrick', 'forestgreen']

    means = [df[df['prompt_type'] == k]['mean_ttft_ms'].mean() for k in keys]
    stds  = [df[df['prompt_type'] == k]['mean_ttft_ms'].std()  for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='white', linewidth=0.8,
                  error_kw=dict(elinewidth=1.5, ecolor='#333333', capthick=1))

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds),
                f'{mean:.1f} ms',
                ha='center', va='bottom', fontsize=9, color='#333333')

    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra=f'ISL={isl}  |  OSL={osl}  |  Concurrency=1')
    ax.set_ylabel('Mean TTFT (ms)', labelpad=8)
    ax.set_ylim(0, max(means) * 1.1)
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    return fig

def plot_prompt_type_e2el(df, gpu_type, isl=512, osl=64, title=None):
    title = title or 'TTFT by Prompt Type'

    df['prompt_type'] = df.apply(get_prompt_type, axis=1)
    df = df[df['prompt_type'].notna()]

    keys   = ['easy', 'hard', 'random']
    labels = ['Easy\n(repetitive)', 'Hard\n(coherent text)', 'Random\n(vLLM random)']
    colors = ['cornflowerblue', 'firebrick', 'forestgreen']

    means = [df[df['prompt_type'] == k]['mean_e2el_ms'].mean() for k in keys]
    stds  = [df[df['prompt_type'] == k]['mean_e2el_ms'].std()  for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='white', linewidth=0.8,
                  error_kw=dict(elinewidth=1.5, ecolor='#333333', capthick=1))

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds),
                f'{mean:.1f} ms',
                ha='center', va='bottom', fontsize=9, color='#333333')

    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra=f'ISL={isl}  |  OSL={osl}  |  Concurrency=1')
    ax.set_ylabel('Mean E2E Latency (ms)', labelpad=8)
    ax.set_ylim(0, max(means) * 1.1)
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    return fig

def gpu_mem_piechart(gpu_type, total_vram, gpu_mem_util=0.85):
    """ gpu memory breakdown pie chart (static allocation) """

    labels = ['Model Weights', 
             'KV Cache \n(reserved)',
             'Peak Activations & \nFramework Overhead', 
             'Unreserved / Buffer']
    colors = ['cornflowerblue', 'indianred', 'olivedrab', 'darkgrey']

    weight_gb = 16.0
    activation_gb = total_vram * 0.1
    reserved_gb = total_vram * gpu_mem_util
    kvcache_gb = reserved_gb - weight_gb - activation_gb
    buffer_gb = total_vram - reserved_gb

    sizes = [weight_gb, kvcache_gb, activation_gb, buffer_gb]

    def autopct_func(pct):
        gb = pct / 100 * total_vram
        gb_str = f"{gb:.0f}" if abs(gb - round(gb)) < 0.1 else f"{gb:.1f}"
        pct_str = f"{pct:.0f}" if abs(pct - round(pct)) < 0.5 else f"{pct:.1f}"
        return f"{pct_str}%\n({gb_str} GB)"
    
    
    fig, ax = plt.subplots(figsize=(7,5))

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct=autopct_func,
           colors=colors, startangle=120, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
           textprops={'fontsize': 10})
    for at in autotexts:
        at.set_fontsize(9)
    
    ax.set_title('GPU Memory Usage Breakdown', fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='BF16 | Static Allocation')
    plt.tight_layout()
    return fig


def plot_ttft_zero_mean(df, gpu_type, run_number='0', prompt_type='easy', title=None):
    title = title or 'Zero-Mean TTFT Across Sequential Identical Prompts'

    df = df.copy()

    df['ttfts'] = df['ttfts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df['prompt_type'] = df.apply(get_prompt_type, axis=1)
    df = df[df['prompt_type'].notna()]
    df = df[(df['prompt_type'] == prompt_type)].reset_index(drop=True)

    df_exploded = df.explode('ttfts')
    df_exploded['ttfts'] = df_exploded['ttfts'].astype(float)
    ttfts =  df.loc[run_number, 'ttfts']
    ttfts = np.array(ttfts, dtype=float)

    ttft_zero_mean = ttfts - np.mean(ttfts)
    x = np.arange(len(ttft_zero_mean))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, ttft_zero_mean)
    ax.axhline(0, linewidth=1)

    ax.set_title(title, fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', extra=f'Concurrency = 1 | Prompt Type: {prompt_type}')
    ax.set_ylabel('Zero-mean TTFT (ms)', labelpad=8)
    ax.set_xlabel('Prompt Index')
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    return fig


def plot_memory_vs_concurrency(gpu_type='A100 40GB', total_vram=40, gpu_mem_util=0.85, isl=512, osl=512, concurrency_levels=None):
    """ analytical gpu memory as a function of concurrency """

    if concurrency_levels is None:
        concurrency_levels = [1, 4, 6, 8, 12, 16, 24, 32, 48, 64, 
                               80, 96, 112, 128, 144, 160, 192, 256, 320, 384, 448, 512]

    # llama 3.1 8B constants
    num_layers  = 32
    num_kv_heads = 8
    head_dim = 128
    dtype_bytes = 2 # bf16 (2 bytes)
    kv_factor   = 2 # keys and values

    avg_tokens = isl + osl / 2
    kv_per_request_gb = (num_layers * avg_tokens * num_kv_heads * head_dim * kv_factor * dtype_bytes) / 1e9

    weight_gb       = 16.0
    activation_gb   = total_vram * 0.05
    reserved_ceiling = total_vram * gpu_mem_util
    kv_pool_gb      = reserved_ceiling - weight_gb - activation_gb

    conc = np.array(concurrency_levels)
    kv_used_gb   = conc * kv_per_request_gb
    total_used_gb = weight_gb + activation_gb + kv_used_gb

    # saturation point: where KV cache usage exceeds reserved pool
    saturation_conc = kv_pool_gb / kv_per_request_gb

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.fill_between(conc, 0, weight_gb,
                    alpha=0.7, color='cornflowerblue', label=f'Model Weights ({weight_gb:.0f} GB)')
    ax.fill_between(conc, weight_gb, weight_gb + activation_gb,
                    alpha=0.7, color='olivedrab', label=f'Activations & Framework ({activation_gb:.0f} GB)')
    ax.fill_between(conc, weight_gb + activation_gb, total_used_gb,
                    alpha=0.7, color='indianred', label='KV Cache (active requests)')

    # ceiling lines
    ax.axhline(reserved_ceiling, color='black', linewidth=1.5, linestyle='--',
               label=f'Reserved ceiling ({reserved_ceiling:.0f} GB, util={gpu_mem_util})')

    # saturation annotation
    ax.axvline(saturation_conc, color='#E85C4C', linewidth=1.2, linestyle=':')
    ax.annotate(f'KV cache\nsaturation\n≈{saturation_conc:.0f} requests',
                xy=(saturation_conc, reserved_ceiling * 0.75),
                xytext=(saturation_conc + 20, reserved_ceiling * 0.75),
                fontsize=8.5,
                arrowprops=dict(arrowstyle='->', lw=1.2))

    ax.set_xlim(conc[0], conc[-1])
    ax.set_ylim(0, total_vram)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(conc)
    ax.tick_params(axis='x', rotation=45)

    ax.set_title('GPU Memory Usage vs Concurrency', fontweight='bold', pad=20)
    subtitle(ax, gpu_type, model='Llama-3.1-8B', 
             extra=f'ISL / OSL = {isl} / {osl}  |  GPU Memory Util = 0.85 | Analytical estimate')
    ax.set_xlabel('Concurrency (max in-flight requests)', labelpad=8)
    ax.set_ylabel('GPU Memory (GB)', labelpad=8)
    ax.legend(fontsize=9, loc='lower left')
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
plot_itl_boxplot(parquet_path = itldistributions_path, 
                 gpu_type = 'A100 40GB')


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

#%% [markdown] 
# ## V100 Concurrency Sweep 
data = v100_concurrency_data
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
plot_itl_boxplot(parquet_path = itldistributions_path, 
                 gpu_type = 'V100 32GB')


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
fig = plot_metrics_vs_concurrency_comparison(a100_df=a100_concurrency_data,
                                             v100_df=v100_concurrency_data,
                                             metric='mean_itl_ms',
                                             ylabel='ITL (ms)',
                                             title='GPU Comparison: Inter-Token Latency vs Concurrency')
fig = plot_metrics_vs_concurrency_comparison(a100_df=a100_concurrency_data,
                                             v100_df=v100_concurrency_data,
                                             metric='mean_tpot_ms',
                                             ylabel='TPOT (ms)',
                                             title='GPU Comparison: Time Per Output Token (TPOT) vs Concurrency')
fig = plot_metrics_vs_concurrency_comparison(a100_df=a100_concurrency_data,
                                             v100_df=v100_concurrency_data,
                                             metric='output_throughput',
                                             ylabel='Output Throughput (tokens/s)',
                                             title='GPU Comparison: Output Throughput vs Concurrency')
fig = plot_metrics_vs_concurrency_comparison(a100_df=a100_concurrency_data,
                                             v100_df=v100_concurrency_data,
                                             metric='mean_ttft_ms',
                                             ylabel='TTFT (ms)',
                                             title='GPU Comparison: Time to First Token (ms) vs Concurrency')
plot_mean_vs_p99_comparison(a100_df=a100_concurrency_data,
                            v100_df=v100_concurrency_data,
                            metric='itl')
plot_mean_vs_p99_comparison(a100_df=a100_concurrency_data,
                            v100_df=v100_concurrency_data,
                            metric='tpot')
plot_itl_boxplot_comparison(parquet_path=itldistributions_path)

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

# %%
fig = plot_prompt_type_ttft(prompttype_df, gpu_type='A100 40GB', isl=512, osl=64)
fig = plot_prompt_type_e2el(prompttype_df, gpu_type='A100 40GB', isl=512, osl=64, title="E2E Latency by Prompt Type")
# %%
fig = plot_ttft_zero_mean(prompttype_df_2, gpu_type='A100 40GB', run_number=0, prompt_type='easy')

fig = plot_ttft_zero_mean(prompttype_df_2, gpu_type='A100 40GB', run_number=1, prompt_type='easy')


fig = plot_ttft_zero_mean(prompttype_df_2, gpu_type='A100 40GB', run_number=0, prompt_type='hard')

fig = plot_ttft_zero_mean(prompttype_df_2, gpu_type='A100 40GB', run_number=1, prompt_type='hard')


fig = plot_ttft_zero_mean(prompttype_df_2, gpu_type='A100 40GB', run_number=0, prompt_type='random')

fig = plot_ttft_zero_mean(prompttype_df_2, gpu_type='A100 40GB', run_number=1, prompt_type='random')
# %%


fig = gpu_mem_piechart(gpu_type='A100 40GB',
                       total_vram=40)
fig = gpu_mem_piechart(gpu_type='V100 32GB',
                       total_vram=32)

fig = plot_memory_vs_concurrency(gpu_type='A100 40GB',
                           total_vram=40)
fig = plot_memory_vs_concurrency(gpu_type='V100 32GB',
                           total_vram=32)
# %%
plot_itl_boxplot(parquet_path = itldistributions_path, 
                 gpu_type = 'A100 40GB',
                 showfliers=True)

plot_itl_boxplot(parquet_path = itldistributions_path, 
                 gpu_type = 'A100 40GB',
                 showfliers=False)

plot_itl_boxplot_outliers(parquet_path = itldistributions_path, 
                 gpu_type = 'A100 40GB')
# %%
