#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

from utils.shared_config import *
from utils.plotting import *

set_figure_style()


def plot_metric_vs_metric(
    series, df, x_metric, y_metric, 
    xlabel, ylabel, title,
    saturation_lines=None,
    y_lim=None,
    subtitle_info=None,
    paper=False,
):
    """ line plot of metric vs concurrency for a list of series entries """
    fig, ax = plt.subplots()

    for s in series:
        # sort values before plotting
        plot_df = filter_df(df, **s['filters']).sort_values(x_metric)
        ax.plot(
            plot_df[x_metric],
            plot_df[y_metric],
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
 
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
#     ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    pass
    #%%
    results_dir = "../results/"

    # load data
    power_df = pd.read_csv(f"{results_dir}concurrency-sweep/power_metrics.csv")
    power_df = power_df[(power_df['num_power_samples'] > 100)]

    a100_8b_default = build_tp_series('a100', 'llama-3.1-8b')
    a100_8b_p2pdisabled = build_tp_series('a100', 'llama-3.1-8b', 'p2p_disabled')
    a100_13b_default = build_tp_series('a100', 'llama-2-13b')
    a100_13b_p2pdisabled = build_tp_series('a100', 'llama-2-13b', 'p2p_disabled')

    sub_a100_8b = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')
    sub_a100_13b = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.1-13B', 
           config='ISL/OSL = 512/512')
    
    fig = plot_metric_vs_concurrency(a100_13b_default, power_df, metric='joules_per_output_token',
                                     ylabel="Energy per Output Token (J)", title='Energy Usage Across Concurrencies')
    fig = plot_metric_vs_concurrency(a100_13b_p2pdisabled, power_df, metric='joules_per_output_token',
                                     ylabel="Energy per Output Token (J)", title='Energy Usage Across Concurrencies: \nP2P Disabled')
    
    fig = plot_metric_vs_concurrency(a100_13b_default, power_df, metric='joules_per_total_token',
                                     ylabel="Energy per Total Token (J)", title='Energy Usage Across Concurrencies')
    fig = plot_metric_vs_concurrency(a100_13b_p2pdisabled, power_df, metric='joules_per_total_token',
                                     ylabel="Energy per Total Token (J)", title='Energy Usage Across Concurrencies: \nP2P Disabled')
    

    fig = plot_metric_vs_metric(a100_13b_default, power_df, 
                                x_metric="median_itl_ms", y_metric="joules_per_output_token",
                                xlabel="Median ITL (ms)", ylabel="Energy per Token (J)",
                                title='Energy Usage Across TP Degrees')
    fig = plot_metric_vs_metric(a100_13b_p2pdisabled, power_df, 
                                x_metric="median_itl_ms", y_metric="joules_per_output_token",
                                xlabel="Median ITL (ms)", ylabel="Energy per Token (J)",
                                title='Energy Usage Across TP Degrees')
    


# %%
