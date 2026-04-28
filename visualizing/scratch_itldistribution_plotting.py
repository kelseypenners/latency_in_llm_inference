#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import ast

from plot_utils import *

# load itl data
itldistributions_path = '../results/single-gpu/concurrency-sweep/fine-grained/itl_distributions.parquet'

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
    bp_bot = ax_bot.boxplot(data, tick_labels=concurrency_levels,
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
        subtitle(ax_top, gpu_type=gpu_type, model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')

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
    bp = ax.boxplot(data, tick_labels=concurrency_levels,
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
        subtitle(ax, gpu_type=gpu_type, model='Llama-3.1-8B', extra='ISL / OSL = 512 / 512')
    ax.set_xlabel('Concurrency (max in-flight requests)', labelpad=6)
    ax.set_ylabel('ITL (ms)', labelpad=6)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
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

        bp = ax.boxplot(data, tick_labels=concurrency_levels, showfliers=False, patch_artist=True)
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

#%%
fig = plot_itl_boxplot(parquet_path=itldistributions_path, 
                       gpu_type='A100 40GB', paper=True)
fig = plot_itl_boxplot(parquet_path=itldistributions_path, 
                       gpu_type='V100 32GB', paper=True)

fig = plot_itl_boxplot(parquet_path=itldistributions_path,
                       gpu_type='A100 40GB', showfliers=True)
fig = plot_itl_boxplot(parquet_path=itldistributions_path,
                       gpu_type='A100 40GB', showfliers=False)
fig = plot_itl_boxplot_outliers(parquet_path=itldistributions_path,
                                gpu_type='A100 40GB')