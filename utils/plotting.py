import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import matplotlib as mpl

import seaborn as sns

def subtitle(ax, **fields):
    text = '  |  '.join(str(val) for val in fields.values())
    ax.text(0.5, 1.03, text, transform=ax.transAxes,
            ha='center', fontsize=8, color='gray')
    
def save_fig(fig, filename, output_dir = "./"):
    output_dir = Path(output_dir)
    fig.savefig(f'{output_dir}/{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"saved: {filename}")

def get_prompt_type(row):
    path = str(row.get('dataset-path', '') or '')
    if 'easy' in path:
        return 'easy'
    elif 'hard' in path:
        return 'hard'
    elif str(row.get('dataset-name', '')).lower() == 'random':
        return 'random'
    return None

def filter_df(df, **filters):
    """ filter data frame given filters {column: value, ...} """
    data = df.copy()

    for col, val in filters.items():
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            data = data[data[col].isin(val)]
        else:
            data = data[data[col] == val]

    return data

def set_figure_style():

    sns.set_theme(style="whitegrid")

    mpl.rcParams.update({
        # figure size
        "figure.figsize": (3.5, 2.5),
        "figure.dpi": 300,

        # fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "text.usetex": False,

        # font sizes
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,

        # lines
        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # grid
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        # save fig
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.07,
    })

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