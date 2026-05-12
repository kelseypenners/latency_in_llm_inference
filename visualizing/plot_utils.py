import matplotlib.pyplot as plt
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
