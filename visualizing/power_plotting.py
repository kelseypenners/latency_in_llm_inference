#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import matplot2tikz

from utils.shared_config import *
from utils.plotting import *

set_figure_style()


def plot_percent_energy_increase(
    series_default, series_nop2p, df, x_metric, y_metric,
    xlabel, title, series_socket=None, plot_tp1=False,
    subtitle_info=None, paper=False
):
    """ plots percent increase in energy cost compared to default baseline """

    fig, ax = plt.subplots()
    if series_socket is None:
        zip_groups = zip(series_default, [[s] for s in series_nop2p])
    else:
        zip_groups = zip(series_default, [[n, s] for n, s in zip(series_nop2p, series_socket)])
    
    for s_def, comparison_series in zip_groups:
        df_def = filter_df(df, **s_def['filters']).sort_values(x_metric)
        if df_def.empty:
            continue
        if s_def['filters']['tp'] == 1:
            continue

        for s_comp in comparison_series:
            df_comp = filter_df(df, **s_comp['filters']).sort_values(x_metric)

            merged = pd.merge(df_def, df_comp, on=x_metric, suffixes=('_def', '_comp'))
            if merged.empty:
                continue
                
            # calculate percentage overhead relative to default
            merged['overhead_pct'] = ((merged[f'{y_metric}_comp'] - merged[f'{y_metric}_def']) / merged[f'{y_metric}_def']) * 100

            ax.plot(
                merged[x_metric],
                merged['overhead_pct'],
                marker=s_comp.get('marker', s_def['marker']),
                markersize=1.5,
                label=s_comp.get('label', s_def['label']),
                color=s_comp.get('color', s_def['color']),
                linestyle=s_comp.get('linestyle', '-')
            )

    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy Increase (%)")
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
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
    a100_8b_socket = build_tp_series('a100', 'llama-3.1-8b', 'network_socket')

    a100_13b_default = build_tp_series('a100', 'llama-2-13b')
    a100_13b_p2pdisabled = build_tp_series('a100', 'llama-2-13b', 'p2p_disabled')

    a100_1b_default = build_tp_series('a100', 'llama-3.2-1b')
    a100_1b_p2pdisabled = build_tp_series('a100', 'llama-3.2-1b', 'p2p_disabled')

    
    for i in range(len(a100_8b_p2pdisabled)):
        for model_config in [a100_8b_p2pdisabled, a100_1b_p2pdisabled, a100_13b_p2pdisabled]:
            config = model_config[i]
            config['linestyle'] = "-"
            config['label'] = config['label'] + " (SHM)"
    for config in a100_8b_socket:
        config["linestyle"] = "-."
        config["label"] = config["label"] + " (Socket)"

    a100_8b_combined = a100_8b_default + a100_8b_p2pdisabled + a100_8b_socket
    a100_13b_combined = a100_13b_default + a100_13b_p2pdisabled
    a100_1b_combined = a100_1b_default + a100_1b_p2pdisabled

    sub_a100_8b = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')
    sub_a100_1b = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.2-1B', 
           config='ISL/OSL = 512/512')
    sub_a100_13b = dict(gpu='shy-fec (A100s)', 
           model='Llama-2-13B', 
           config='ISL/OSL = 512/512')
    
    #%%
    
    # fig = plot_metric_vs_concurrency(a100_8b_combined, power_df, metric='joules_per_output_token',
    #                                  ylabel="Energy per Output Token (J)", title='Energy Usage Across Concurrencies: \nP2P / P2P Disabled',
    #                                  subtitle_info=sub_a100_8b)
    # fig = plot_metric_vs_concurrency(a100_1b_combined, power_df, metric='joules_per_output_token',
    #                                  ylabel="Energy per Output Token (J)", title='Energy Usage Across Concurrencies: \nP2P / P2P Disabled',
    #                                  subtitle_info=sub_a100_1b)
    # fig = plot_metric_vs_concurrency(a100_13b_combined, power_df, metric='joules_per_output_token',
    #                                  ylabel="Energy per Output Token (J)", title='Energy Usage Across Concurrencies: \nP2P / P2P Disabled',
    #                                  subtitle_info=sub_a100_13b)

    fig = plot_metric_vs_concurrency(a100_8b_combined, power_df, metric='joules_per_output_token',
                                     ylabel="Energy per Token (J)", title='Energy Usage Across Concurrencies',
                                     subtitle_info=sub_a100_8b)
    matplot2tikz.save("idk.tex")
    # fig = plot_metric_vs_concurrency(a100_1b_combined, power_df, metric='joules_per_total_token',
    #                                  ylabel="Energy per Total Token (J)", title='Energy Usage Across Concurrencies: \nP2P / P2P Disabled',
    #                                  subtitle_info=sub_a100_1b)
    # fig = plot_metric_vs_concurrency(a100_13b_combined, power_df, metric='joules_per_total_token',
    #                                  ylabel="Energy per Total Token (J)", title='Energy Usage Across Concurrencies: \nP2P / P2P Disabled',
    #                                  subtitle_info=sub_a100_13b)
    
    # median itl plots
    # fig = plot_metric_vs_metric(a100_8b_combined, power_df, 
    #                             x_metric="mean_itl_ms", y_metric="joules_per_output_token",
    #                             xlabel="Median ITL (ms)", ylabel="Energy per Token (J)",
    #                             title='Energy Usage Across TP Degrees', subtitle_info=sub_a100_8b)
    # fig = plot_metric_vs_metric(a100_1b_combined, power_df, 
    #                             x_metric="mean_itl_ms", y_metric="joules_per_output_token",
    #                             xlabel="Median ITL (ms)", ylabel="Energy per Token (J)",
    #                             title='Energy Usage Across TP Degrees', subtitle_info=sub_a100_1b)
    # fig = plot_metric_vs_metric(a100_13b_combined, power_df, 
    #                             x_metric="mean_itl_ms", y_metric="joules_per_output_token",
    #                             xlabel="Median ITL (ms)", ylabel="Energy per Token (J)",
    #                             title='Energy Usage Across TP Degrees', subtitle_info=sub_a100_13b)
    

    #%%

    fig1 = plot_percent_energy_increase(
        a100_8b_default, a100_8b_p2pdisabled, power_df, series_socket=a100_8b_socket, 
        x_metric="max_concurrency", y_metric="joules_per_output_token",
        xlabel="Concurrency", 
        title="Percent Energy Increase per Token Across Transports",
        subtitle_info=sub_a100_8b
    )
    matplot2tikz.save("idk1.tex")

    fig2 = plot_percent_energy_increase(
        a100_13b_default, a100_13b_p2pdisabled, power_df, 
        x_metric="max_concurrency", y_metric="joules_per_total_token",
        xlabel="Concurrency", 
        title="Percent Energy Increase per Token when Disabling P2P",
        subtitle_info=sub_a100_13b
    )
# %%
