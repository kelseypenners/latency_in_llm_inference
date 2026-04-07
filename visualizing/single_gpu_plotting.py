#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import ast

from plot_utils import *

# load sequence length sweep data
seqlen_df = pd.read_csv('../results/single-gpu/seqlen-sweep/averaged_single_gpu_seqlen_sweep_results.csv')
# load baseline concurrency sweep data
concurrency_df = pd.read_csv('../results/single-gpu/concurrency-sweep/baseline/averaged_concurrency_sweep_baseline_results.csv')
# load fine-grained concurrency sweep data
concurrency_df = pd.read_csv('../results/single-gpu/concurrency-sweep/fine-grained/averaged_concurrency_sweep_fine_grained_results.csv')
# load itl data
itldistributions_path = '../results/single-gpu/concurrency-sweep/fine-grained/itl_distributions.parquet'
# load prompt type data
prompttype_df = pd.read_csv('../results/single-gpu/dataset-comparison/prompttype_10runs_1prompt.csv')
# load prompt type data
prompttype_df_2 = pd.read_csv('../results/single-gpu/dataset-comparison/prompttype_prefixcache_results.csv')

# gpu-specific
a100_seqlen_data = seqlen_df[seqlen_df['gpu_type'] == 'a100'].copy()
v100_seqlen_data = seqlen_df[seqlen_df['gpu_type'] == 'v100'].copy()

a100_concurrency_data = concurrency_df[concurrency_df['gpu_type'] == 'a100'].copy()
v100_concurrency_data = concurrency_df[concurrency_df['gpu_type'] == 'v100'].copy()

set_figure_style()

# SEQLEN SWEEP PLOTS ----------------------------------------------------
def plot_input_output_heatmap(df, gpu_type, values, title, cmap='Oranges', paper=False):
    """ heatmap of metric across all ISL/OSL configs """

    heatmap_lens = [128, 256, 512, 1024, 2048]
    df = df[df['random_input_len'].isin(heatmap_lens)]
    pivot = df.pivot_table(values=values,
                           index='random_output_len', 
                           columns='random_input_len')

    fig, ax = plt.subplots()
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap=cmap, ax=ax,
                linecolor='white', linewidths=0.5, annot_kws={"size": 7})
    ax.invert_yaxis()
    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='Concurrency = 1')
    ax.set_xlabel('Input Sequence Length (tokens)', labelpad=6)
    ax.set_ylabel('Output Sequence Length (tokens)', labelpad=6)
    plt.tight_layout()

    return fig

def plot_ttft_vs_isl(df, gpu_type, osl=None, title=None, paper=False):
    """ TTFT vs ISL, one line per OSL value (or a fixed OSL) """

    if osl is None:
        osl_values = sorted(df['random_output_len'].unique())
    else:
        osl_values = [osl]
    title = title or 'TTFT Scales Linearly with ISL, Independent of OSL'

    fig, ax = plt.subplots()
    for osl_val in osl_values:
        data = df[df['random_output_len'] == osl_val]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'],
                marker='o', label=f"OSL={osl_val}")
    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='Concurrency = 1')
    ax.set_xlabel('Input Sequence Length (ISL) (tokens)')
    ax.legend(title='Output Length', loc='upper left')
    ax.set_ylabel('TTFT (ms)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def plot_ttft_vs_osl_per_isl(df, paper=False):
    """ multiple plots: TTFT vs OSL for each ISL """

    isl_values = sorted(df['random_input_len'].unique())
    fig, axes = plt.subplots(1, len(isl_values),
                             figsize=(6, 2.8), sharey=True)

    for ax, isl in zip(axes, isl_values):
        data = df[df['random_input_len'] == isl]
        ax.plot(data['random_output_len'], data['mean_ttft_ms'], marker='o')
        ax.set_title(f'ISL={isl}', pad=6)
        ax.set_xlabel('OSL (tokens)')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('TTFT (ms)')
    plt.suptitle('TTFT vs OSL (fixed ISL)', y=1.01)
    plt.tight_layout()

    return fig

def plot_ttft_vs_isl_per_osl(df, paper=False):
    """ multiple plots: TTFT vs ISL for each OSL """

    osl_values = sorted(df['random_output_len'].unique())
    fig, axes = plt.subplots(1, len(osl_values),
                             figsize=(6, 2.8), sharey=True)

    for ax, osl in zip(axes, osl_values):
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'], marker='o')
        ax.set_title(f'OSL={osl}', pad=6)
        ax.set_xlabel('ISL (tokens)')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('TTFT (ms)')
    plt.suptitle('TTFT vs ISL (fixed OSL)', y=1.01)
    plt.tight_layout()

    return fig

def plot_e2el_vs_osl(df, gpu_type, isl=None, title=None, paper=False):
    """ E2E latency vs OSL, one line per ISL (or fixed ISL) """

    if isl is None:
        isl_values = sorted(df['random_input_len'].unique())
    else:
        isl_values = [isl]
    title = title or 'Impact of Output Sequence Length (OSL) on E2E Latency'

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for isl_val in isl_values:
        data = df[df['random_input_len'] == isl_val]
        ax.plot(data['random_output_len'], data['mean_e2el_ms'],
                marker='o', label=f"ISL={isl_val}")
    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='Concurrency = 1')
    ax.set_xlabel('OSL (tokens)')
    ax.legend(title='Input Length')
    ax.set_ylabel('E2E Latency (ms)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def plot_e2el_vs_isl_per_osl(df, paper=False):
    """ multiple plots: E2E latency vs ISL for each OSL """

    osl_values = sorted(df['random_output_len'].unique())
    title = 'E2E Latency vs ISL (fixed OSL)'

    fig, axes = plt.subplots(1, len(osl_values),
                             figsize=(6, 2.8), sharey=True)
    for ax, osl in zip(axes, osl_values):
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_e2el_ms'], marker='o')
        ax.set_title(f'OSL={osl}', pad=6)
        ax.set_xlabel('ISL (tokens)')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('E2E Latency (ms)')
    plt.suptitle(title, y=1.01)
    plt.tight_layout()

    return fig

# CONCURRENCY SWEEP PLOTS ----------------------------------------------------
def plot_metrics_vs_concurrency(df, gpu_type, metric, label, title, paper=False):
    """ metric versus max concurrency """

    fig, ax = plt.subplots()
    ax.plot(df['max_concurrency'], df[metric], marker='o', color='#4C9BE8')
    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency (max in-flight requests)')
    ax.set_ylabel(label)
    ax.set_xticks(df['max_concurrency'])
    #ax.set_xscale('log', base=2)
    #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig

def plot_mean_vs_p99(df, gpu_type, metric="itl", title=None, p90=True, paper=False):
    """ mean vs P99 for ITL vs concurrency """

    title = title or f'{metric.upper()}: Mean vs Tail Latency vs Concurrency'

    fig, ax = plt.subplots()
    ax.plot(df['max_concurrency'], df[f'mean_{metric}_ms'],
            marker='o', label='Mean', color='#4C9BE8')
    ax.plot(df['max_concurrency'], df[f'p99_{metric}_ms'],
            marker='s', linestyle='--', label='P99', color='#E85C4C')
    ax.plot(df['max_concurrency'], df[f'p90_{metric}_ms'],
            marker='s', linestyle='--', label='P90', color='#E8A84C')
    ax.fill_between(df['max_concurrency'],
                    df[f'mean_{metric}_ms'], df[f'p99_{metric}_ms'],
                    alpha=0.1, color='#4C9BE8', label='Mean--P99 gap')

    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency', labelpad=6)
    ax.set_ylabel(f'{metric.upper()} (ms)', labelpad=6)
    ax.set_xticks(df['max_concurrency'])
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig

def plot_itl_boxplot(parquet_path, gpu_type, title=None, showfliers=False, paper=False):
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
    data = [df_exploded[df_exploded['max_concurrency'] == c]['itls_ms'].values
            for c in concurrency_levels]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bp = ax.boxplot(data, labels=concurrency_levels,
                    showfliers=showfliers, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#4C9BE8')
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('#E85C4C')
        median.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_color('#444444')
        whisker.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color('#444444')
        cap.set_linewidth(1.0)

    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency (max in-flight requests)', labelpad=6)
    ax.set_ylabel('ITL (ms)', labelpad=6)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    return fig

def plot_itl_boxplot_outliers(parquet_path, gpu_type, title=None, paper=False):
    """
    ITL boxplot with broken y-axis.
    Bottom panel: full box/whisker without fliers, zoomed to bulk distribution.
    Top panel: log-scale outlier scatter only, showing extreme tail values.
    """
    title = title or 'ITL Distribution vs Concurrency (with Outliers)'

    df = pd.read_parquet(parquet_path)
    if gpu_type.lower().replace(' ', '') in ['a10040gb', 'a100']:
        df = df[df['gpu_type'] == 'a100']
    else:
        df = df[df['gpu_type'] == 'v100']

    df_exploded = df.explode('itls_ms')
    df_exploded['itls_ms'] = df_exploded['itls_ms'].astype(float)

    concurrency_levels = sorted(df_exploded['max_concurrency'].unique())
    data = [df_exploded[df_exploded['max_concurrency'] == c]['itls_ms'].values
            for c in concurrency_levels]

    # compute break point just above max whisker
    whisker_tops = []
    for d in data:
        q25, q75 = np.percentile(d, [25, 75])
        iqr = q75 - q25
        whisker_tops.append(min(q75 + 1.5 * iqr, np.max(d)))
    break_point = max(whisker_tops) * 1.15

    all_vals = np.concatenate(data)
    outlier_max = np.percentile(all_vals, 99.5)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5.5, 5.0),
        gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.08}
    )

    box_kwargs = dict(
        patch_artist=True,
        medianprops=dict(color='#E85C4C', linewidth=1.5),
        whiskerprops=dict(color='#444444', linewidth=1.0),
        capprops=dict(color='#444444', linewidth=1.0),
        boxprops=dict(linewidth=0.8),
    )

    # bottom panel: boxes + whiskers, no fliers
    bp_bot = ax_bot.boxplot(data, labels=concurrency_levels,
                            showfliers=False, **box_kwargs)
    for patch in bp_bot['boxes']:
        patch.set_facecolor('#4C9BE8')
        patch.set_alpha(0.7)
    ax_bot.set_ylim(0, break_point * 0.92)

    # top panel: log-scale outlier scatter only
    ax_top.set_yscale('log')
    for i, (c, d) in enumerate(zip(concurrency_levels, data), start=1):
        outliers = d[d > break_point]
        if len(outliers) > 0:
            jitter = np.random.uniform(-0.3, 0.3, size=len(outliers))
            ax_top.scatter(np.full_like(outliers, i) + jitter, outliers,
                           color='#E85C4C', alpha=0.5, s=8, linewidths=0, zorder=3)
    ax_top.set_ylim(break_point, outlier_max * 1.5)

    from matplotlib.ticker import LogLocator, LogFormatter
    ax_top.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax_top.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=True))

    n = len(concurrency_levels)
    ax_bot.set_xlim(0.5, n + 0.5)
    ax_top.set_xlim(0.5, n + 0.5)
    ax_top.set_xticks([])

    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['top'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False, which='both')

    d_mark = 0.012
    break_kwargs = dict(color='#555555', linewidth=1.0, clip_on=False)
    ax_top.plot((-d_mark, +d_mark), (-d_mark, +d_mark),
                transform=ax_top.transAxes, **break_kwargs)
    ax_top.plot((1 - d_mark, 1 + d_mark), (-d_mark, +d_mark),
                transform=ax_top.transAxes, **break_kwargs)
    ax_bot.plot((-d_mark, +d_mark), (1 - d_mark, 1 + d_mark),
                transform=ax_bot.transAxes, **break_kwargs)
    ax_bot.plot((1 - d_mark, 1 + d_mark), (1 - d_mark, 1 + d_mark),
                transform=ax_bot.transAxes, **break_kwargs)

    ax_top.text(0.01, 0.95, f'Outliers only (> {break_point:.0f} ms, log scale)',
                transform=ax_top.transAxes, fontsize=7,
                color='#E85C4C', va='top')
    ax_bot.text(0.01, 0.97, f'Bulk distribution ($\\leq$ {break_point:.0f} ms)',
                transform=ax_bot.transAxes, fontsize=7,
                color='#444444', va='top')

    ax_top.set_ylabel('ITL (ms)', labelpad=6)
    ax_bot.set_ylabel('ITL (ms)', labelpad=6)
    ax_bot.set_xlabel('Concurrency (max in-flight requests)', labelpad=6)
    ax_bot.tick_params(axis='x', rotation=45)
    ax_top.grid(True, axis='y', alpha=0.3)
    ax_bot.grid(True, axis='y', alpha=0.3)

    fig.suptitle(title, y=0.98)
    if not paper:
        subtitle(ax_top, gpu_type, model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')

    plt.tight_layout()
    return fig

# COMPARISON PLOTS ----------------------------------------------------
def plot_run_variance(seqlen_df, concurrency_df, gpu_type, stds=1, osl=512, paper=False):
    """ TTFT run variance (seqlen) and throughput run variance (concurrency) """

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))

    # TTFT variance
    ax = axes[0]
    fixed_osl = seqlen_df[seqlen_df['random_output_len'] == osl]
    ax.plot(fixed_osl['random_input_len'], fixed_osl['mean_ttft_ms'],
            label='Mean TTFT', color='#4C9BE8')
    if 'std_runs_mean_ttft_ms' in fixed_osl.columns:
        ax.fill_between(fixed_osl['random_input_len'],
                        fixed_osl['mean_ttft_ms'] - stds * fixed_osl['std_runs_mean_ttft_ms'],
                        fixed_osl['mean_ttft_ms'] + stds * fixed_osl['std_runs_mean_ttft_ms'],
                        alpha=0.2, color='#4C9BE8', label=f'$\\pm${stds} std')

    ax.set_xlabel('Input Sequence Length (tokens)', labelpad=6)
    ax.set_ylabel('TTFT (ms)', labelpad=6)
    ax.set_title('TTFT: Run-to-Run Variance', pad=8)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B', extra=f'OSL={osl}, Concurrency=1')
    ax.set_xticks(sorted(fixed_osl['random_input_len'].unique()))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(concurrency_df['max_concurrency'], concurrency_df['output_throughput'],
            label='Mean Throughput', color='#4C9BE8')
    if 'std_runs_output_throughput' in concurrency_df.columns:
        ax.fill_between(concurrency_df['max_concurrency'],
                        concurrency_df['output_throughput'] - stds * concurrency_df['std_runs_output_throughput'],
                        concurrency_df['output_throughput'] + stds * concurrency_df['std_runs_output_throughput'],
                        alpha=0.2, color='#4C9BE8', label=f'$\\pm${stds} std')

    ax.set_xlabel('Concurrency (max in-flight requests)', labelpad=6)
    ax.set_ylabel('Output Throughput (tokens/s)', labelpad=6)
    ax.set_title('Throughput: Run-to-Run Variance', pad=8)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='ISL/OSL = 512/512')
    ax.set_xticks(concurrency_df['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7)

    plt.tight_layout()
    return fig

def plot_heatmap_comparison(a100_df, v100_df, values, title, cmap='Oranges',
                             same_range=False, paper=False):
    """ comparison of heatmaps between GPUs for same metric """

    heatmap_lens = [128, 256, 512, 1024, 2048]
    a100_df = a100_df[a100_df['random_input_len'].isin(heatmap_lens)]
    v100_df = v100_df[v100_df['random_input_len'].isin(heatmap_lens)]
    
    dfs = [a100_df, v100_df]
    gpu_types = ['A100 40GB', 'V100 32GB']
    pivots = [df.pivot_table(values=values, index='random_output_len', columns='random_input_len')
              for df in dfs]
    if same_range:
        vmin = min(p.min().min() for p in pivots)
        vmax = max(p.max().max() for p in pivots)
    else:
        vmin = vmax = None

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.3))
    for ax, pivot, gtype in zip(axes, pivots, gpu_types):
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                    linecolor='white', linewidths=0.5,
                    annot_kws={"size": 7}, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(gtype, fontsize=9, pad=8)
        ax.set_xlabel('ISL (tokens)', labelpad=6)
        ax.set_ylabel('OSL (tokens)', labelpad=6)

    fig.suptitle(title, fontsize=10, y=0.99)
    if not paper:
        fig.text(0.5, 0.97, 'Llama-3.1-8B  |  Concurrency = 1',
                 ha='center', va='top', fontsize=7, color='grey')
    plt.tight_layout()

    return fig

def plot_ttft_vs_isl_comparison(a100_df, v100_df, osl=512, title=None, paper=False):
    """ TTFT vs ISL comparison between GPUs """

    title = title or 'TTFT vs ISL -- A100 vs V100'

    labels = ['A100 40GB', 'V100 32GB']
    dfs = [a100_df, v100_df]
    colors = ['#4C9BE8', '#E85C4C']

    fig, ax = plt.subplots()
    for df, label, color in zip(dfs, labels, colors):
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'],
                marker='o', label=label, color=color)
    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, gpu_type='A100 40GB vs V100 32GB', model='Llama-3.1-8B',
                 extra=f'Concurrency = 1  | OSL={osl}')
    ax.set_xlabel('Input Sequence Length (ISL) (tokens)')
    ax.legend(title='GPU Type', loc='upper left')
    ax.set_ylabel('TTFT (ms)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def plot_metrics_vs_concurrency_comparison(a100_df, v100_df, metric, ylabel, title, paper=False):
    """ metric versus max concurrency comparison between GPUs """

    labels = ['A100', 'V100']
    dfs = [a100_df, v100_df]
    colors = ['#4C9BE8', '#E85C4C']

    fig, ax = plt.subplots()
    for df, label, color in zip(dfs, labels, colors):
        ax.plot(df['max_concurrency'], df[metric],
                marker='o', label=label, color=color)
    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, 'A100 vs V100', model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_xticks(dfs[-1]['max_concurrency'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(title='GPU Type', loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig

def plot_mean_vs_p99_comparison(a100_df, v100_df, metric='itl', title=None, paper=False):
    """ mean vs P99 comparison between GPUs """

    title = title or f'{metric.upper()}: Mean vs Tail Latency -- A100 vs V100'

    dfs = [a100_df, v100_df]
    gpu_types = ['A100 40GB', 'V100 32GB']

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2), sharey=True)
    for ax, df, gtype in zip(axes, dfs, gpu_types):
        ax.plot(df['max_concurrency'], df[f'mean_{metric}_ms'],
                marker='o', label='Mean', color='#4C9BE8')
        ax.plot(df['max_concurrency'], df[f'p99_{metric}_ms'],
                marker='s', linestyle='--', label='P99', color='#E85C4C')
        ax.plot(df['max_concurrency'], df[f'p90_{metric}_ms'],
                marker='s', linestyle='--', label='P90', color='#E8A84C')
        ax.fill_between(df['max_concurrency'],
                        df[f'mean_{metric}_ms'], df[f'p99_{metric}_ms'],
                        alpha=0.1, color='#4C9BE8')
        ax.set_title(gtype, pad=8)
        ax.set_xlabel('Concurrency', labelpad=6)
        ax.set_ylabel(f'{metric.upper()} (ms)', labelpad=6)
        ax.set_xticks(df['max_concurrency'])
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(title, y=1.01)
    if not paper:
        fig.text(0.5, 0.97, 'Llama-3.1-8B  |  ISL / OSL = 512 / 512',
                 ha='center', va='top', fontsize=7, color='grey')
    plt.tight_layout()

    return fig

def plot_itl_boxplot_comparison(parquet_path, title=None, paper=False):
    """ ITL distribution box-whisker comparison between A100 and V100 """

    title = title or 'ITL Distribution vs Concurrency -- A100 vs V100'

    df_all = pd.read_parquet(parquet_path)
    gpu_types = ['a100', 'v100']
    gpu_labels = ['A100 40GB', 'V100 32GB']

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2), sharey=True)
    for ax, gpu, label in zip(axes, gpu_types, gpu_labels):
        df = df_all[df_all['gpu_type'] == gpu]
        df_exploded = df.explode('itls_ms')
        df_exploded['itls_ms'] = df_exploded['itls_ms'].astype(float)

        concurrency_levels = sorted(df_exploded['max_concurrency'].unique())
        data = [df_exploded[df_exploded['max_concurrency'] == c]['itls_ms'].values
                for c in concurrency_levels]

        bp = ax.boxplot(data, labels=concurrency_levels, showfliers=False, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#4C9BE8')
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('#E85C4C')
            median.set_linewidth(1.5)
        for whisker in bp['whiskers']:
            whisker.set_color('#444444')
            whisker.set_linewidth(1.0)
        for cap in bp['caps']:
            cap.set_color('#444444')
            cap.set_linewidth(1.0)
        ax.set_title(label, pad=8)
        ax.set_xlabel('Concurrency (max in-flight requests)', labelpad=6)
        ax.set_ylabel('ITL (ms)', labelpad=6)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(title, y=1.01)
    if not paper:
        fig.text(0.5, 0.97, 'Llama-3.1-8B  |  ISL / OSL = 512 / 512',
                 ha='center', va='top', fontsize=7, color='grey')
    plt.tight_layout()

    return fig

def plot_kv_saturation_comparison(a100_df, v100_df, title=None, paper=False):
    """ comparison of KV cache saturation effect on throughput and ITL across GPUs """

    title = title or 'KV Cache Saturation'

    configs = [
        {'df': a100_df, 'gpu_label': 'A100 40GB', 'sat_concurrency': 139,
         'thp_color': '#4C9BE8', 'itl_color': '#E85C4C'},
        {'df': v100_df, 'gpu_label': 'V100 32GB', 'sat_concurrency': 79,
         'thp_color': '#4C9BE8', 'itl_color': '#E85C4C'},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax_left, cfg in zip(axes, configs):
        df        = cfg['df']
        gpu_label = cfg['gpu_label']
        sat_c     = cfg['sat_concurrency']
        thp_color = cfg['thp_color']
        itl_color = cfg['itl_color']

        ax_right = ax_left.twinx()

        l1, = ax_left.plot(df['max_concurrency'], df['output_throughput'],
                           color=thp_color, marker='o', label='Output Throughput')
        ax_left.set_ylabel('Output Throughput (tokens/s)', color=thp_color, labelpad=6)
        ax_left.tick_params(axis='y', labelcolor=thp_color)

        l2, = ax_right.plot(df['max_concurrency'], df['mean_itl_ms'],
                            color=itl_color, marker='s', label='Mean ITL')
        ax_right.set_ylabel('Mean ITL (ms)', color=itl_color, labelpad=6)
        ax_right.tick_params(axis='y', labelcolor=itl_color)

        ax_left.axvline(x=sat_c, color='green', linestyle=':', linewidth=1.2, zorder=3)
        ax_left.axvspan(sat_c, df['max_concurrency'].max(),
                        alpha=0.07, color='#E85C4C', zorder=0)

        thp_min = df['output_throughput'].min()
        thp_max = df['output_throughput'].max()
        # ax_left.annotate(
        #     f'KV cache\nsaturation\n$\\approx${sat_c}',
        #     xy=(sat_c, ann_y),
        #     xytext=(sat_c * 1.15, ann_y),
        #     fontsize=7, color='grey', va='center',
        #     arrowprops=dict(arrowstyle='->', color='grey', lw=1.0),
        # )

        ax_left.set_xscale('log', base=2)
        ax_left.set_xticks([4, 8, 16, 32, 64, 128, 256, 512])
        ax_left.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax_left.tick_params(axis='x', rotation=45)
        ax_left.set_xlabel('Concurrency', labelpad=6)
        ax_left.grid(True, axis='y', alpha=0.3)
        ax_left.set_axisbelow(True)
        ax_left.set_title(gpu_label, fontsize=9, pad=8)
        ax_left.legend([l1, l2], [l1.get_label(), l2.get_label()],
                       loc='upper left', fontsize=7)

    fig.suptitle(title, fontsize=10, y=1.01)
    if not paper:
        fig.text(0.5, 0.97, 'Single GPU  |  Llama-3.1-8B  |  ISL / OSL = 512 / 512',
                 ha='center', va='top', fontsize=7, color='grey')
    plt.tight_layout()

    return fig


# DATASET COMPARISON PLOTS ----------------------------------------------------
def plot_prompt_type_ttft(df, gpu_type, isl=512, osl=64, title=None, paper=False):
    title = title or 'TTFT by Prompt Type'

    df = df[df['enable-prefix-caching']==False]
    keys   = ['easy', 'hard', 'random']
    labels = ['Easy\n(repetitive)', 'Hard\n(coherent text)', 'Random\n(vLLM random)']
    colors = ['#4C9BE8', '#E85C4C', '#5CB85C']

    means = [df[df['prompt_type'] == k]['mean_ttft_ms'].mean() for k in keys]
    stds  = [df[df['prompt_type'] == k]['mean_ttft_ms'].std()  for k in keys]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors, alpha=0.8, edgecolor='white', linewidth=0.6,
                  error_kw=dict(elinewidth=1.2, ecolor='#333333', capthick=1))

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds),
                f'{mean:.1f} ms',
                ha='center', va='bottom', fontsize=7, color='#333333')

    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B',
                 extra=f'ISL={isl}  |  OSL={osl}  |  Concurrency=1')
    ax.set_ylabel('Mean TTFT (ms)', labelpad=6)
    ax.set_ylim(0, max(means) * 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig

def plot_prompt_type_e2el(df, gpu_type, isl=512, osl=64, title=None, paper=False):
    title = title or 'E2E Latency by Prompt Type'

    keys   = ['easy', 'hard', 'random']
    labels = ['Easy\n(repetitive)', 'Hard\n(coherent text)', 'Random\n(vLLM random)']
    colors = ['#4C9BE8', '#E85C4C', '#5CB85C']

    means = [df[df['prompt_type'] == k]['mean_e2el_ms'].mean() for k in keys]
    stds  = [df[df['prompt_type'] == k]['mean_e2el_ms'].std()  for k in keys]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors, alpha=0.8, edgecolor='white', linewidth=0.6,
                  error_kw=dict(elinewidth=1.2, ecolor='#333333', capthick=1))

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds),
                f'{mean:.1f} ms',
                ha='center', va='bottom', fontsize=7, color='#333333')

    ax.set_title(title, pad=10)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B',
                 extra=f'ISL={isl}  |  OSL={osl}  |  Concurrency=1')
    ax.set_ylabel('Mean E2E Latency (ms)', labelpad=6)
    ax.set_ylim(0, max(means) * 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig

def gpu_mem_piechart(gpu_type=None, total_vram=None, gpu_mem_util=0.85, title=None, paper=False):
    """ GPU memory breakdown pie chart (static allocation) """

    labels = ['Model Weights',
              'KV Cache\n(reserved)',
              'Peak Activations &\nFramework Overhead',
              'Unreserved / Buffer']
    colors = ['#4C9BE8', '#E85C4C', '#5CB85C', '#AAAAAA']

    def get_sizes(total_vram):
        weight_gb     = 16.0
        activation_gb = total_vram * 0.1
        reserved_gb   = total_vram * gpu_mem_util
        kvcache_gb    = reserved_gb - weight_gb - activation_gb
        buffer_gb     = total_vram - reserved_gb
        return [weight_gb, kvcache_gb, activation_gb, buffer_gb]

    def make_autopct(total_vram):
        def autopct_func(pct):
            gb      = pct / 100 * total_vram
            gb_str  = f"{gb:.0f}" if abs(gb - round(gb)) < 0.1 else f"{gb:.1f}"
            pct_str = f"{pct:.0f}" if abs(pct - round(pct)) < 0.5 else f"{pct:.1f}"
            return f'{pct_str}%\n({gb_str} GB)'
        return autopct_func

    if gpu_type is None:
        # side-by-side A100 / V100
        configs = [('A100 40GB', 40), ('V100 32GB', 32)]
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.5))
        for ax, (gtype, vram) in zip(axes, configs):
            wedges, texts, autotexts = ax.pie(
                get_sizes(vram), labels=labels, autopct=make_autopct(vram),
                colors=colors, startangle=120,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
                textprops={'fontsize': 7}
            )
            for at in autotexts:
                at.set_fontsize(7)
            ax.set_title(gtype, pad=8)
        fig.suptitle('GPU Memory Usage Breakdown', y=1.01)
        if not paper:
            fig.text(0.5, 0.97, 'Llama-3.1-8B  |  BF16  |  Static Allocation',
                     ha='center', va='top', fontsize=7, color='grey')
    else:
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            get_sizes(total_vram), labels=labels, autopct=make_autopct(total_vram),
            colors=colors, startangle=120,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
            textprops={'fontsize': 7}
        )
        for at in autotexts:
            at.set_fontsize(7)
        #ax.set_title(title, pad=10)
        if not paper:
            subtitle(ax, gpu_type, model='Llama-3.1-8B', extra='BF16 | Static Allocation')

    plt.tight_layout()
    return fig

def plot_ttft_raw(df, gpu_type, run_number=0, prompt_type=None, title=None, paper=False):
    title = title or 'TTFT Across Sequential Identical Prompts'

    df = df.copy()
    df['ttfts'] = df['ttfts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df = df[df['prompt_type'].notna()]

    prompt_configs = [
        ('easy',   '#4C9BE8'),
        ('hard',   '#E85C4C'),
        ('random', '#5CB85C'),
    ]
    cache_configs = [
        (True,  'ON',  'solid'),
        (False, 'OFF', 'dashed'),
    ]

    fig, ax = plt.subplots()

    if prompt_type is not None:
        subset = df[df['prompt_type'] == prompt_type].reset_index(drop=True)
        color = next(c for p, c in prompt_configs if p == prompt_type)
        for enabled, cache_label, linestyle in cache_configs:
            rows = subset[subset['enable-prefix-caching'] == enabled].reset_index(drop=True)
            ttfts = np.array(rows.loc[run_number, 'ttfts'], dtype=float) * 1000
            ax.plot(np.arange(len(ttfts)), ttfts,
                    label=f'Prefix caching {cache_label}',
                    color=color, linestyle=linestyle,
                    alpha=0.8, marker='o', markersize=2.5)
        if not paper:
            subtitle(ax, gpu_type, model='Llama-3.1-8B',
                     extra=f'Concurrency = 1 | Prompt Type: {prompt_type}')
        ax.legend(fontsize=7)
    else:
        for p_type, color in prompt_configs:
            subset = df[df['prompt_type'] == p_type].reset_index(drop=True)
            for enabled, cache_label, linestyle in cache_configs:
                rows = subset[subset['enable-prefix-caching'] == enabled].reset_index(drop=True)
                ttfts = np.array(rows.loc[run_number, 'ttfts'], dtype=float) * 1000
                ax.plot(np.arange(len(ttfts)), ttfts,
                        label=f'{p_type} -- caching {cache_label}',
                        color=color, linestyle=linestyle,
                        alpha=0.8, marker='o', markersize=2.5)
        if not paper:
            subtitle(ax, gpu_type, model='Llama-3.1-8B',
                     extra='Concurrency = 1 | Prompt Types: repetitive, coherent, random')

        legend_handles = [
            plt.Line2D([0], [0], color=color, linewidth=1.5, label=p_type)
            for p_type, color in prompt_configs
        ]
        ax.legend(title='Prompt type', handles=legend_handles, fontsize=7, loc='best')

        x_text = 40
        ax.text(x_text + 1, 23, 'prefix caching ON',
                fontsize=7, color='grey', va='center', ha='left')
        ax.text(x_text + 1, 39, 'prefix caching OFF',
                fontsize=7, color='grey', va='center', ha='left')

    ax.set_title(title, pad=10)
    ax.set_ylabel('TTFT (ms)', labelpad=6)
    ax.set_xlabel('Prompt Index')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def plot_ttft_zero_mean(df, gpu_type, run_number=0, prompt_type=None,
                        prefix_caching=True, title=None, paper=False):
    """ zero-mean scatter plot of TTFTs across 100 sequential prompts """

    title = title or 'Zero-Mean TTFT Across Sequential Identical Prompts'

    df = df.copy()
    df['ttfts'] = df['ttfts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df = df[df['prompt_type'].notna()]

    fig, ax = plt.subplots()

    if prompt_type is not None:
        df = df[df['prompt_type'] == prompt_type].reset_index(drop=True)
        subset = df[df['enable-prefix-caching'] == prefix_caching].reset_index(drop=True)
        ttfts = np.array(subset.loc[run_number, 'ttfts'], dtype=float) * 1000
        ttft_zero_mean = ttfts - np.mean(ttfts)
        ax.scatter(np.arange(len(ttft_zero_mean)), ttft_zero_mean,
                   color='#4C9BE8', alpha=0.7, s=12)
        if not paper:
            subtitle(ax, gpu_type, model='Llama-3.1-8B',
                     extra=f'Concurrency = 1 | Prompt Type: {prompt_type}')
    else:
        prompt_configs = [('easy',   '#4C9BE8'),
                          ('hard',   '#E85C4C'),
                          ('random', '#5CB85C')]
        df = df[df['enable-prefix-caching'] == prefix_caching].reset_index(drop=True)
        for p_type, color in prompt_configs:
            subset = df[df['prompt_type'] == p_type].reset_index(drop=True)
            ttfts = np.array(subset.loc[run_number, 'ttfts'], dtype=float) * 1000
            ttft_zero_mean = ttfts - np.mean(ttfts)
            print(f'{p_type} mean = {np.mean(ttfts)}')
            ax.scatter(np.arange(len(ttft_zero_mean)), ttft_zero_mean,
                       label=p_type, color=color, alpha=0.7, s=12)
        ax.legend(title='Prompt Type', fontsize=7, loc='lower left')
        if not paper:
            subtitle(ax, gpu_type, model='Llama-3.1-8B',
                     extra=f'Concurrency = 1 | Enable prefix caching: {prefix_caching}')

    ax.axhline(0, color='#444444', linewidth=0.8)
    ax.set_title(title, pad=10)
    ax.set_ylabel('Zero-mean TTFT (ms)', labelpad=6)
    ax.set_xlabel('Prompt Index')
    ax.set_ylim(-3, 3)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig

def plot_memory_vs_concurrency(gpu_type='A100 40GB', total_vram=40, gpu_mem_util=0.85,
                                isl=512, osl=512, concurrency_levels=None, paper=False):
    """ analytical GPU memory as a function of concurrency """

    if concurrency_levels is None:
        concurrency_levels = [1, 4, 6, 8, 12, 16, 24, 32, 48, 64,
                               80, 96, 112, 128, 144, 160, 192, 256, 320, 384, 448, 512]

    # Llama-3.1-8B constants
    num_layers   = 32
    num_kv_heads = 8
    head_dim     = 128
    dtype_bytes  = 2  # bf16
    kv_factor    = 2  # keys and values

    avg_tokens = isl + osl / 2
    kv_per_request_gb = (num_layers * avg_tokens * num_kv_heads * head_dim * kv_factor * dtype_bytes) / 1e9

    weight_gb        = 16.0
    activation_gb    = total_vram * 0.1
    reserved_ceiling = total_vram * gpu_mem_util
    kv_pool_gb       = reserved_ceiling - weight_gb - activation_gb

    conc          = np.array(concurrency_levels)
    kv_used_gb    = conc * kv_per_request_gb
    total_used_gb = weight_gb + activation_gb + kv_used_gb
    saturation_conc = kv_pool_gb / kv_per_request_gb

    fig, ax = plt.subplots()

    ax.fill_between(conc, 0, weight_gb,
                    alpha=0.75, color='#4C9BE8',
                    label=f'Model Weights ({weight_gb:.0f} GB)')
    ax.fill_between(conc, weight_gb, weight_gb + activation_gb,
                    alpha=0.75, color='#5CB85C',
                    label=f'Activations & Framework ({activation_gb:.0f} GB)')
    ax.fill_between(conc, weight_gb + activation_gb, total_used_gb,
                    alpha=0.75, color='#E85C4C',
                    label='KV Cache (active requests)')

    ax.axhline(reserved_ceiling, color='#333333', linewidth=1.2, linestyle='--',
               label=f'Reserved ceiling ({reserved_ceiling:.0f} GB, util={gpu_mem_util})')

    ax.axvline(saturation_conc, color='#E85C4C', linewidth=1.0, linestyle=':')
    ax.annotate(f'KV cache\nsaturation\n$\\approx${saturation_conc:.0f} req.',
                xy=(saturation_conc, reserved_ceiling * 0.75),
                xytext=(saturation_conc + 20, reserved_ceiling * 0.75),
                fontsize=7,
                arrowprops=dict(arrowstyle='->', lw=1.0))

    ax.set_xlim(conc[0], conc[-1])
    ax.set_ylim(0, total_vram)
    ax.set_xscale('log', base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='x', rotation=45)

    ax.set_title('GPU Memory Usage vs Concurrency', pad=10)
    if not paper:
        subtitle(ax, gpu_type, model='Llama-3.1-8B',
                 extra=f'ISL / OSL = {isl} / {osl}  |  GPU Mem Util = 0.85 | Analytical')
    ax.set_xlabel('Concurrency (max in-flight requests)', labelpad=6)
    ax.set_ylabel('GPU Memory (GB)', labelpad=6)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


# A100 PLOTS ----------------------------------------------------
#%% [markdown]
# ## A100 Sequence Length Sweep
data = a100_seqlen_data
gpu_type = 'A100 40GB'
fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
                                values='mean_ttft_ms',
                                title='A100 TTFT [ms]',
                                cmap='coolwarm',
                                paper=True)
save_fig(fig, 'a100_ttft')
# fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
#                                 values='output_throughput',
#                                 title='A100 Output Throughput [tokens/s]',
#                                 cmap='coolwarm_r',
#                                 paper=True)
# save_fig(fig, 'a100_throughput')
# fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
#                                 values='mean_itl_ms',
#                                 title='A100 Mean ITL [ms]',
#                                 cmap='coolwarm',
#                                 paper=True)
# save_fig(fig, 'a100_itl')
# fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
#                                 values='total_token_throughput',
#                                 title='Total Token Throughput (tokens/s)',
#                                 cmap='coolwarm_r',
#                                 paper=True)
# fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
#                                 values='mean_e2el_ms',
#                                 title='A100 E2EL [ms]',
#                                 cmap='coolwarm',
#                                 paper=True)
# save_fig(fig, 'a100_e2el')
fig = plot_ttft_vs_isl(df=data, gpu_type=gpu_type, paper=True)
# fig = plot_e2el_vs_osl(df=data, gpu_type=gpu_type, paper=True)

#%% [markdown]
# ## A100 Concurrency Sweep
data = a100_concurrency_data
fig = plot_metrics_vs_concurrency(df=data, gpu_type=gpu_type,
                                  metric='output_throughput',
                                  label='Output Throughput (tokens/s)',
                                  title='Output Throughput vs Concurrency',
                                  paper=True)
fig = plot_metrics_vs_concurrency(df=data, gpu_type=gpu_type,
                                  metric='mean_itl_ms',
                                  label='ITL (ms)',
                                  title='ITL (ms) vs Concurrency',
                                  paper=True)
fig = plot_metrics_vs_concurrency(df=data, gpu_type=gpu_type,
                                  metric='mean_ttft_ms',
                                  label='TTFT (ms)',
                                  title='TTFT (ms) vs Concurrency',
                                  paper=True)
fig = plot_metrics_vs_concurrency(df=data, gpu_type=gpu_type,
                                  metric='mean_tpot_ms',
                                  label='TPOT (ms)',
                                  title='TPOT (ms) vs Concurrency',
                                  paper=True)
fig = plot_mean_vs_p99(df=data, gpu_type=gpu_type, metric='itl', paper=True)
fig = plot_mean_vs_p99(df=data, gpu_type=gpu_type, metric='tpot', paper=True)
fig = plot_itl_boxplot(parquet_path=itldistributions_path, 
                       gpu_type='A100 40GB', paper=True)


# V100 PLOTS ----------------------------------------------------
#%% [markdown]
# ## V100 Sequence Length Sweep
data = v100_seqlen_data
gpu_type = 'V100 32GB'
# fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
#                                 values='mean_ttft_ms',
#                                 title='V100 TTFT [ms]',
#                                 cmap='coolwarm',
#                                 paper=True)
# save_fig(fig, 'v100_ttft')
# fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
#                                 values='output_throughput',
#                                 title='V100 Output Throughput [tokens/s]',
#                                 cmap='coolwarm_r',
#                                 paper=True)
# save_fig(fig, 'v100_throughput')
# fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
#                                 values='mean_itl_ms',
#                                 title='V100 Mean ITL [ms]',
#                                 cmap='coolwarm',
#                                 paper=True)
# save_fig(fig, 'v100_itl')
# fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
#                                 values='total_token_throughput',
#                                 title='Total Token Throughput (tokens/s)',
#                                 cmap='coolwarm_r',
#                                 paper=True)
# fig = plot_input_output_heatmap(df=data, gpu_type=gpu_type,
#                                 values='mean_e2el_ms',
#                                 title='V100 E2EL [ms]',
#                                 cmap='coolwarm',
#                                 paper=True)
# save_fig(fig, 'v100_e2el')
# fig = plot_ttft_vs_isl(df=data, gpu_type=gpu_type, paper=True)
# fig = plot_e2el_vs_osl(df=data, gpu_type=gpu_type, paper=True)

#%% [markdown]
# ## V100 Concurrency Sweep
data = v100_concurrency_data
fig = plot_metrics_vs_concurrency(df=data, gpu_type=gpu_type,
                                  metric='output_throughput',
                                  label='Output Throughput (tokens/s)',
                                  title='Output Throughput vs Concurrency',
                                  paper=True)
fig = plot_metrics_vs_concurrency(df=data, gpu_type=gpu_type,
                                  metric='mean_itl_ms',
                                  label='ITL (ms)',
                                  title='ITL (ms) vs Concurrency',
                                  paper=True)
fig = plot_metrics_vs_concurrency(df=data, gpu_type=gpu_type,
                                  metric='mean_ttft_ms',
                                  label='TTFT (ms)',
                                  title='TTFT (ms) vs Concurrency',
                                  paper=True)
fig = plot_metrics_vs_concurrency(df=data, gpu_type=gpu_type,
                                  metric='mean_tpot_ms',
                                  label='TPOT (ms)',
                                  title='TPOT (ms) vs Concurrency',
                                  paper=True)
fig = plot_mean_vs_p99(df=data, gpu_type=gpu_type, metric='itl', paper=True)
fig = plot_mean_vs_p99(df=data, gpu_type=gpu_type, metric='tpot', paper=True)
fig = plot_itl_boxplot(parquet_path=itldistributions_path, 
                       gpu_type='V100 32GB', paper=True)


# COMPARISON ----------------------------------------------------
#%% [markdown]
# ## A100 vs V100 Comparison
fig = plot_ttft_vs_isl_comparison(a100_df=a100_seqlen_data,
                                  v100_df=v100_seqlen_data,
                                  osl=512,
                                  paper=True)
fig = plot_heatmap_comparison(a100_df=a100_seqlen_data, v100_df=v100_seqlen_data,
                              values='mean_ttft_ms',
                              title='Time to First Token (TTFT) [ms]',
                              cmap='coolwarm',
                              paper=True)
fig = plot_heatmap_comparison(a100_df=a100_seqlen_data, v100_df=v100_seqlen_data,
                              values='output_throughput',
                              title='Output Throughput (tokens/s)',
                              cmap='coolwarm_r',
                              paper=True)
fig = plot_heatmap_comparison(a100_df=a100_seqlen_data, v100_df=v100_seqlen_data,
                              values='mean_itl_ms',
                              title='Inter-Token Latency (ITL) (ms)',
                              cmap='coolwarm',
                              paper=True)
fig = plot_metrics_vs_concurrency_comparison(a100_df=a100_concurrency_data,
                                             v100_df=v100_concurrency_data,
                                             metric='mean_itl_ms',
                                             ylabel='ITL (ms)',
                                             title='GPU Comparison: Inter-Token Latency vs Concurrency',
                                             paper=True)
fig = plot_metrics_vs_concurrency_comparison(a100_df=a100_concurrency_data,
                                             v100_df=v100_concurrency_data,
                                             metric='mean_tpot_ms',
                                             ylabel='TPOT (ms)',
                                             title='GPU Comparison: TPOT vs Concurrency',
                                             paper=True)
# fig = plot_metrics_vs_concurrency_comparison(a100_df=a100_concurrency_data,
#                                              v100_df=v100_concurrency_data,
#                                              metric='output_throughput',
#                                              ylabel='Output Throughput (tokens/s)',
#                                              title='Output Throughput vs Concurrency',
#                                              paper=True)
# save_fig(fig, 'throughputconcurrency')
# fig = plot_metrics_vs_concurrency_comparison(a100_df=a100_concurrency_data,
#                                              v100_df=v100_concurrency_data,
#                                              metric='mean_ttft_ms',
#                                              ylabel='TTFT [ms]',
#                                              title='TTFT vs Concurrency',
#                                              paper=True)
# save_fig(fig, 'ttftconcurrency')
# fig = plot_kv_saturation_comparison(a100_df=a100_concurrency_data,
#                                     v100_df=v100_concurrency_data,
#                                     paper=True)
# save_fig(fig, 'kvsaturationplot')

# RUN VARIANCE ----------------------------------------------------
# # %% [markdown]
# ## Run Variance
# seqlen_data = a100_seqlen_data
# concurrency_data = a100_concurrency_data
# gpu_type = 'A100 40GB'
# fig = plot_run_variance(seqlen_data, concurrency_data, gpu_type=gpu_type, stds=1, osl=512)

# seqlen_data = v100_seqlen_data
# concurrency_data = a100_concurrency_data
# gpu_type = 'V100 32GB'
# fig = plot_run_variance(seqlen_data, concurrency_data, gpu_type=gpu_type, stds=1, osl=512)

# PROMPT TYPE PLOTS ----------------------------------------------------
# %% [markdown]
## Prompt Type
fig = plot_prompt_type_ttft(prompttype_df_2, gpu_type='A100 40GB', isl=512, osl=64, paper=True)
save_fig(fig, 'prompttype_ttft_means')
#fig = plot_prompt_type_e2el(prompttype_df_2, gpu_type='A100 40GB', isl=512, osl=64, paper=True)

# %%
#fig = plot_ttft_zero_mean(prompttype_df_2, gpu_type='A100 40GB', run_number=1, prefix_caching=True)
fig = plot_ttft_zero_mean(prompttype_df_2, gpu_type='A100 40GB', run_number=1, prefix_caching=False, paper=True)
save_fig(fig, 'prompttype_ttft_zeromean')
fig = plot_ttft_raw(prompttype_df_2, gpu_type='A100 40GB', run_number=1, paper=True)
save_fig(fig, 'prompttype_prefixcaching')
# %%
# fig = gpu_mem_piechart(gpu_type='A100 40GB', total_vram=40, 
#                        title='A100 GPU Memory Allocation', paper=True)
# save_fig(fig, 'a100_gpupie')
# fig = gpu_mem_piechart(gpu_type='V100 32GB', total_vram=32, 
#                        title='V100 GPU Memory Allocation', paper=True)
# save_fig(fig, 'v100_gpupie')

# fig = gpu_mem_piechart(paper=True)

fig = plot_memory_vs_concurrency(gpu_type='A100 40GB', total_vram=40, paper=True)
fig = plot_memory_vs_concurrency(gpu_type='V100 32GB', total_vram=32, paper=True)

# %%
# fig = plot_itl_boxplot(parquet_path=itldistributions_path,
#                        gpu_type='A100 40GB', showfliers=True)
# fig = plot_itl_boxplot(parquet_path=itldistributions_path,
#                        gpu_type='A100 40GB', showfliers=False)
# fig = plot_itl_boxplot_outliers(parquet_path=itldistributions_path,
#                                 gpu_type='A100 40GB')
# %%