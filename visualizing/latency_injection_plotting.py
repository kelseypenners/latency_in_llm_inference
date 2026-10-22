#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
import matplot2tikz

from utils.shared_config import *
from utils.plotting import *

set_figure_style()

def validate_injection_deltas(series, df, model, metric='median_itl_ms', metric_label='ITL'):

    model = MODELS[model]
    num_layers = model['num_layers']

    # compute for each concurrency in series list
    for s in series:
        plot_df = filter_df(df, **s['filters']).sort_values("injected_latency_us")

        # get nvlink baseline (no added latency)
        nvlink_baseline_rows = plot_df[plot_df["injected_latency_us"] == 0]
        baseline = nvlink_baseline_rows[metric].values[0]

        print(f"\nValidation for {model['model_name']} | {s['label']}")
        print(f"    baseline {metric_label} (no injection): {baseline:.2f} ms")
        print(f"{'Injected (us)':<15}{'Expected Δ (ms)':<17}{'Actual Δ (ms)':<15}{'Error (ms)':<16}{'Percent Error':<15}")
        print("-" * 82)
        
        for _, row in plot_df.iterrows():
            injected_us = row["injected_latency_us"]
            actual = row[metric]
            
            # calculate expected and actual change
            expected_delta_ms = (injected_us / 1000.0) * num_layers * 2
            actual_delta_ms = actual - baseline
            
            # calculate error
            error_ms = actual_delta_ms - expected_delta_ms
            
            if expected_delta_ms != 0:
                pct_error = (error_ms / expected_delta_ms) * 100
                pct_error_str = f"{pct_error:+.2f}%"
            else:
                pct_error = 0.0
                pct_error_str = "Baseline"
                
            print(f"{injected_us:<15.0f}{expected_delta_ms:<17.0f}{actual_delta_ms:<15.3f}{error_ms:<16.3f}{pct_error_str:<15}")

def plot_validation(series, df, model, metric='median_itl_ms', metric_label='ITL',
                     mode='deltas', subtitle_info=None):
    """ plot injected latency vs expected and observed changes for validation """

    fig, ax = plt.subplots()

    expected_plotted = False

    # get num transformer layers from model config
    model = MODELS[model]
    num_layers = model['num_layers']

    for s in series:
        plot_df = filter_df(df, **s['filters']).sort_values("injected_latency_us")
        
        # get nvlink baseline (no added latency)
        baseline_rows = plot_df[plot_df["injected_latency_us"] == 0]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows[metric].values[0]
        
        # compute deltas
        injected_us = plot_df["injected_latency_us"]
        actual_delta_ms = plot_df[metric] - baseline
        expected_delta_ms = (injected_us / 1000.0) * num_layers * 2


        if mode == 'errors':
            # compute errors
            error_ms = actual_delta_ms - expected_delta_ms

            ax.plot(
                injected_us, 
                error_ms, 
                marker=s.get('marker', 'o'),
                markersize=4,
                label=f"Error ({s['label']})", 
                color=s.get('color'),
                linestyle=s.get('linestyle', '-'),
            )

        elif mode == "deltas":
            # plot expected line once
            if not expected_plotted:
                ax.plot(
                    injected_us, 
                    expected_delta_ms, 
                    label=r'Expected', 
                    color='black', 
                    linestyle='--', 
                    linewidth=1.5,
                    zorder=1
                )
                expected_plotted = True
                
            # plot observed data
            ax.plot(
                injected_us, 
                actual_delta_ms, 
                marker=s.get('marker', 'o'),
                markersize=4,
                label=f"Observed\n({s['label']})", 
                color=s.get('color'),
                linestyle=s.get('linestyle', '-'),
                zorder=2
            )
    if mode == "errors":
        ax.set_title(fr"Mechanism Validation: Errors in $\Delta$ {metric_label}", pad=15)
        ax.set_ylabel(fr"Error (Actual $\Delta$ - Expected $\Delta$ (ms)")
    else:
        ax.set_title(fr"Mechanism Validation: Expected vs Observed $\Delta$ {metric_label}", pad=15)
        ax.set_ylabel(fr"$\Delta$ {metric_label} (ms)")

    ax.set_xlabel(r"Injected Latency ($\mu\text{s}$)")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    if subtitle_info:
        subtitle(ax, **subtitle_info)
    
    plt.tight_layout()
    return fig

def plot_metric_vs_injection(
    series, df, y_metric, ylabel, title,
    per_link_delay=False,
    ns=False,
    y_logscale=False,
    relative=False,
    subtitle_info=None,
    paper=False,
):
    """ line plot of metric vs latency injection for a list of series entries """
    fig, ax = plt.subplots()

    for s in series:
        # sort values before plotting
        plot_df = filter_df(df, **s['filters']).sort_values('injected_latency_us')
        tp = s["filters"]["tp"]

        x_values = plot_df["injected_latency_us"]
        if ns: 
            x_values = x_values * 1000.0
            x_unit = "(ns)"
        else:
            x_unit = r"($\mu$s)"

        if per_link_delay:
            added_delays = x_values / (2 * (tp - 1))
            x_label = fr"Implied Per-Link Delay {x_unit}"
        else:
            added_delays = x_values
            x_label = fr"Injected Latency {x_unit}"
        
        y_values = plot_df[y_metric].copy()

        # plot per gpu output throughput
        if y_metric == "output_throughput":
            y_values = y_values / tp

        if relative:
            # find the baseline row where injected latency == 0
            baseline_val = y_values.iloc[0]
            # percent change
            y_values = ((y_values - baseline_val) / baseline_val) * 100.0

        if ns:
            added_delays = np.where(added_delays <= 0, 0.5, added_delays)
        else:
            added_delays = np.where(added_delays <= 0, 0.0005, added_delays)

        ax.plot(
            added_delays,
            y_values,
            marker=s['marker'],
            markersize=2,
            label=s["label"],
            color=s["color"],
            linestyle=s['linestyle'],
        )
 
    ax.set_title(title, pad=15)
    ax.set_xlabel(x_label)
    if y_metric == "output_throughput":
        ylabel = "Per-GPU " + ylabel
    ax.set_ylabel(ylabel)

    if y_logscale:
        ax.set_yscale('log', base=2)
        ticks_s = [0.1, 1, 10, 60, 300]
        ticks_ms = 1000 * np.array(ticks_s)
        ax.set_yticks(ticks_ms)

        tick_labels = [f"{t/60:g}m" if t > 60 else f"{t}s" for t in ticks_s]
        ax.set_yticklabels(tick_labels)
    if ns:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        #ax.set_xlim(left=0.5)
    else:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, loc='upper left')

    if subtitle_info and not paper:
        subtitle(ax, **subtitle_info)

    plt.tight_layout()
    return fig

def compute_rate_with_added_latency(series, df, metric):
    for s in series:
        filtered_df = filter_df(df, **s['filters']).sort_values('injected_latency_us')
        x = filtered_df['injected_latency_us'].values.reshape(-1,1)
        y = filtered_df[metric].values

        reg = LinearRegression().fit(x,y)
        rate_per_ms = reg.coef_[0] * 1000

        print(f'{s["filters"]["model_name"]} | TP {s["filters"]["tp"]} | {s["label"]}')
        print(f"{metric} increases by {rate_per_ms:.4f} per added ms")


def plot_tp2_to_tp4_extrapolation(df, model_name, concurrency, metric, ylabel, title, subtitle_info=None):

    measured_tp4 = {
        "filters": {
            "gpu_type": "a100", 
            "model_name": model_name, 
            "tp": 4, 
            "max_concurrency": concurrency, 
            "interconnect": "default"
        },
        "label": "TP=4 Measured",
        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-"
    }

    tp2_mask = (
        (df["gpu_type"] == "a100") & 
        (df["model_name"] == model_name) & 
        (df["tp"] == 2) & 
        (df["max_concurrency"] == concurrency)
    )
    
    extrap_df = df.copy()
    
    # apply hop-ratio scaling factor (6 hops / 2 hops)
    extrap_df.loc[tp2_mask, "injected_latency_us"] = extrap_df.loc[tp2_mask, "injected_latency_us"] * 3.0

    predicted_tp4 = {
        "filters": {
            "gpu_type": "a100", 
            "model_name": model_name, 
            "tp": 2,
            "max_concurrency": concurrency, 
            "interconnect": "default"
        },
        "label": "TP=4 Predicted \n(extrapolated from TP=2)",
        "color": "tab:orange",
        "marker": "s",
        "linestyle": "--"
    }

    fig = plot_metric_vs_injection(
        series=[measured_tp4, predicted_tp4],
        df=extrap_df,
        y_metric=metric,
        ylabel=ylabel,
        title=title,
        subtitle_info=subtitle_info
    )
    return fig



if __name__ == "__main__":
    pass
    #%%
    results_dir = "../results/"

    # load data
    power_df = pd.read_csv(f"{results_dir}latency_injection/power_metrics.csv")

    injection_series_tp2_8b = build_latency_series([32,512], gpu_type='a100', model_name='llama-3.1-8b')
    injection_series_tp4_8b = build_latency_series([32,512], gpu_type='a100', 
                                                   model_name='llama-3.1-8b', tp=4)
    injection_series_tp2_13b = build_latency_series([32,512], gpu_type='a100', model_name='llama-2-13b')
    injection_series_tp4_13b = build_latency_series([32,512], gpu_type='a100', 
                                                    model_name='llama-2-13b', tp=4)

    injection_series_tp2_1b = build_latency_series([32,512], gpu_type='a100', model_name='llama-3.2-1b')
    

    injection_series_13b = injection_series_tp2_13b + injection_series_tp4_13b
    injection_series_8b = injection_series_tp2_8b + injection_series_tp4_8b

    for (config_8b, config_13b) in zip(injection_series_8b, injection_series_13b):
        for config in [config_8b, config_13b]:
            tp_config = config["filters"]["tp"]
            config["label"] = f"TP =  {tp_config}, {config["label"]}"

            if tp_config == 4:
                config["linestyle"] = "-."
    
    sub_a100_8b = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.1-8B',
           config='ISL/OSL = 512/512')
    sub_a100_13b = dict(gpu='shy-fec (A100s)', 
           model='Llama-2-13B', 
           config='ISL/OSL = 512/512')
    sub_a100_1b = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.2-1B', 
           tp='TP=2',
           config='ISL/OSL = 512/512')
    
    
    # compute slopes / increase rates
    compute_rate_with_added_latency(injection_series_tp2_8b, power_df, "median_itl_ms")
    compute_rate_with_added_latency(injection_series_tp4_8b, power_df, "median_itl_ms")
    compute_rate_with_added_latency(injection_series_tp2_13b, power_df, "median_itl_ms")
    compute_rate_with_added_latency(injection_series_tp4_13b, power_df, "median_itl_ms")
    compute_rate_with_added_latency(injection_series_tp2_1b, power_df, "median_itl_ms")

    print("")

    compute_rate_with_added_latency(injection_series_tp2_8b, power_df, "joules_per_total_token")
    compute_rate_with_added_latency(injection_series_tp4_8b, power_df, "joules_per_total_token")
    compute_rate_with_added_latency(injection_series_tp2_13b, power_df, "joules_per_total_token")
    compute_rate_with_added_latency(injection_series_tp4_13b, power_df, "joules_per_total_token")
    compute_rate_with_added_latency(injection_series_tp2_1b, power_df, "joules_per_total_token")
    

    #%%
    fig = plot_validation(injection_series_tp2_8b, power_df, "llama_8b", subtitle_info=sub_a100_8b)
    fig = plot_validation(injection_series_tp4_8b, power_df, "llama_8b", subtitle_info=sub_a100_8b)
    fig = plot_validation(injection_series_tp2_8b[:1], power_df, "llama_8b", metric="median_ttft_ms", metric_label="TTFT",
                          mode="errors", subtitle_info=sub_a100_8b)
    fig = plot_validation(injection_series_tp2_8b, power_df, "llama_8b", mode="errors",subtitle_info=sub_a100_8b)
    fig = plot_validation(injection_series_tp4_8b, power_df, "llama_8b", mode="errors",subtitle_info=sub_a100_8b)

    fig = plot_validation(injection_series_tp2_13b, power_df, "llama_13b", subtitle_info=sub_a100_13b)
    fig = plot_validation(injection_series_tp4_13b, power_df, "llama_13b", subtitle_info=sub_a100_13b)
    #fig = plot_validation(injection_series_tp4_13b, power_df, "llama_13b", mode="noterrors", subtitle_info=sub_a100_13b)
    fig = plot_validation(injection_series_tp2_13b, power_df, "llama_13b", mode="errors", subtitle_info=sub_a100_13b)
    fig = plot_validation(injection_series_tp4_13b, power_df, "llama_13b", mode="errors", subtitle_info=sub_a100_13b)
    
    validate_injection_deltas(injection_series_tp2_8b, power_df, 'llama_8b')
    validate_injection_deltas(injection_series_tp4_8b, power_df, 'llama_8b')
    validate_injection_deltas(injection_series_tp2_13b, power_df, "llama_13b")
    validate_injection_deltas(injection_series_tp4_13b, power_df, "llama_13b")

    #%%
    # 8B TP2

    relative=False
    fig = plot_metric_vs_injection(injection_series_tp2_8b, power_df,
                                y_metric="joules_per_total_token",
                                ylabel="Energy per Total Token (J)",
                                ns=True,
                                title="Energy Cost Per Token vs Injected Latency",
                                subtitle_info=sub_a100_8b,
                                relative=relative, per_link_delay=True,)
    fig = plot_metric_vs_injection(injection_series_tp2_8b, power_df,
                                   y_metric="median_itl_ms",
                                   ylabel="Inter-token Latency (ms)",
                                   title="ITL vs Injected Latency",
                                   subtitle_info=sub_a100_8b,
                                   per_link_delay=False)
    fig = plot_metric_vs_injection(injection_series_tp2_8b, power_df,
                                   y_metric="median_ttft_ms",
                                   ylabel="TTFT (ms)",
                                   y_logscale=True,
                                   title="TTFT vs Injected Latency",
                                   subtitle_info=sub_a100_8b)
    fig = plot_metric_vs_injection(injection_series_tp2_8b, power_df,
                                  y_metric="output_throughput",
                                  ylabel="Output Throughput (tokens/s)",
                                  title="Throughput vs Injected Latency",
                                  subtitle_info=sub_a100_8b,
                                  relative=relative)
    
    #%%
    # 8B TP4

    relative=False
    fig = plot_metric_vs_injection(injection_series_tp4_8b, power_df,
                                y_metric="joules_per_output_token",
                                ylabel="Energy per Token (J)",
                                title="Energy Cost Per Token vs Injected Latency",
                                subtitle_info=sub_a100_8b,
                                relative=relative,
                                per_link_delay=True)
    fig = plot_metric_vs_injection(injection_series_tp4_8b, power_df,
                                   y_metric="median_itl_ms",
                                   ylabel="Inter-token Latency (ms)",
                                   title="ITL vs Injected Latency",
                                   subtitle_info=sub_a100_8b,
                                   per_link_delay=False)
    fig = plot_metric_vs_injection(injection_series_tp4_8b, power_df,
                                   y_metric="median_ttft_ms",
                                   ylabel="TTFT (ms)",
                                   y_logscale=True,
                                   title="TTFT vs Injected Latency",
                                   subtitle_info=sub_a100_8b)
    fig = plot_metric_vs_injection(injection_series_tp4_8b, power_df,
                                  y_metric="output_throughput",
                                  ylabel="Output Throughput (tokens/s)",
                                  title="Throughput vs Injected Latency",
                                  subtitle_info=sub_a100_8b,
                                  relative=relative)
    
    #%%
    # 13B TP2
    fig = plot_metric_vs_injection(injection_series_tp2_13b, power_df,
                                y_metric="joules_per_output_token",
                                ylabel="Energy per Token (J)",
                                title="Energy Cost Per Token vs Injected Latency",
                                subtitle_info=sub_a100_13b,
                                relative=relative)
    fig = plot_metric_vs_injection(injection_series_tp2_13b, power_df,
                                   y_metric="median_itl_ms",
                                   ylabel="Inter-token Latency (ms)",
                                   title="ITL vs Injected Latency",
                                   subtitle_info=sub_a100_13b)
    fig = plot_metric_vs_injection(injection_series_tp2_13b, power_df,
                                   y_metric="median_ttft_ms",
                                   ylabel="TTFT (ms)",
                                   y_logscale=True,
                                   title="TTFT vs Injected Latency",
                                   subtitle_info=sub_a100_13b)
    fig = plot_metric_vs_injection(injection_series_tp2_13b, power_df,
                                  y_metric="output_throughput",
                                  ylabel="Output Throughput (tokens/s)",
                                  title="Throughput vs Injected Latency",
                                  subtitle_info=sub_a100_13b,
                                  relative=relative)
    

    #%%
    # 13B TP4
    fig = plot_metric_vs_injection(injection_series_tp4_13b, power_df,
                                y_metric="joules_per_total_token",
                                ylabel="Energy per Total Token (J)",
                                title="Energy Cost Per Token vs Injected Latency",
                                subtitle_info=sub_a100_13b,
                                relative=relative)
    fig = plot_metric_vs_injection(injection_series_tp4_13b, power_df,
                                   y_metric="median_itl_ms",
                                   ylabel="Inter-token Latency (ms)",
                                   title="ITL vs Injected Latency",
                                   subtitle_info=sub_a100_13b)
    fig = plot_metric_vs_injection(injection_series_tp4_13b, power_df,
                                   y_metric="median_ttft_ms",
                                   ylabel="TTFT (ms)",
                                   y_logscale=True,
                                   title="TTFT vs Injected Latency",
                                   subtitle_info=sub_a100_13b)
    fig = plot_metric_vs_injection(injection_series_tp4_13b, power_df,
                                  y_metric="output_throughput",
                                  ylabel="Output Throughput (tokens/s)",
                                  title="Throughput vs Injected Latency",
                                  subtitle_info=sub_a100_13b,
                                  relative=relative)
    
    #%%
    # 1B TP2
    # fig = plot_metric_vs_injection(injection_series_tp2_1b, power_df,
    #                             y_metric="joules_per_total_token",
    #                             ylabel="Energy per Total Token (J)",
    #                             title="Energy Cost Per Token vs Injected Latency",
    #                             subtitle_info=sub_a100_1b,
    #                             relative=relative)
    # fig = plot_metric_vs_injection(injection_series_tp2_1b, power_df,
    #                                y_metric="median_itl_ms",
    #                                ylabel="Inter-token Latency (ms)",
    #                                title="ITL vs Injected Latency",
    #                                subtitle_info=sub_a100_1b)
    # fig = plot_metric_vs_injection(injection_series_tp2_1b, power_df,
    #                                y_metric="median_ttft_ms",
    #                                ylabel="TTFT (ms)",
    #                                y_logscale=True,
    #                                title="TTFT vs Injected Latency",
    #                                subtitle_info=sub_a100_1b)
    # fig = plot_metric_vs_injection(injection_series_tp2_1b, power_df,
    #                               y_metric="output_throughput",
    #                               ylabel="Output Throughput (tokens/s)",
    #                               title="Throughput vs Injected Latency",
    #                               subtitle_info=sub_a100_1b,
    #                               relative=relative)
    #%%
    # all tps for both models

    pld = False
    ns = False
    relative=False
    
    fig = plot_metric_vs_injection(injection_series_8b, power_df,
                                y_metric="joules_per_output_token",
                                ylabel="Energy per Token (J)",
                                title="Energy Cost Per Token vs Injected Latency",
                                subtitle_info=sub_a100_8b,
                                per_link_delay=pld,
                                ns=ns,
                                relative=relative)
    matplot2tikz.save("energycost_injection.tex")
    fig = plot_metric_vs_injection(injection_series_8b, power_df,
                                   y_metric="median_itl_ms",
                                   ylabel="Inter-token Latency (ms)",
                                   title="ITL vs Injected Latency",
                                   subtitle_info=sub_a100_8b,
                                   per_link_delay=pld,
                                   ns=ns)
    matplot2tikz.save("itl_injection.tex")
    fig = plot_metric_vs_injection(injection_series_8b, power_df,
                                   y_metric="median_ttft_ms",
                                   ylabel="TTFT (ms)",
                                   y_logscale=False,
                                   title="TTFT vs Injected Latency",
                                   per_link_delay=pld,
                                   ns=ns,
                                   subtitle_info=sub_a100_8b)
    matplot2tikz.save("ttft_injection.tex")
    fig = plot_metric_vs_injection(injection_series_8b, power_df,
                                  y_metric="output_throughput",
                                  ylabel="Output Throughput (tokens/s)",
                                  title="Throughput vs Injected Latency",
                                  subtitle_info=sub_a100_8b,
                                  per_link_delay=pld, 
                                  ns=ns,
                                  relative=relative,)
    matplot2tikz.save("throughput_injection.tex")


    fig = plot_metric_vs_injection(injection_series_13b, power_df,
                                y_metric="joules_per_output_token",
                                ylabel="Energy per Token (J)",
                                title="Energy Cost Per Token vs Injected Latency",
                                subtitle_info=sub_a100_13b,
                                per_link_delay=pld,
                                ns=ns,
                                relative=False)

    fig = plot_metric_vs_injection(injection_series_13b, power_df,
                                   y_metric="median_itl_ms",
                                   ylabel="Inter-token Latency (ms)",
                                   title="ITL vs Injected Latency",
                                   subtitle_info=sub_a100_13b,
                                   per_link_delay=pld,
                                   ns=ns)
    
    fig = plot_metric_vs_injection(injection_series_13b, power_df,
                                   y_metric="median_ttft_ms",
                                   ylabel="TTFT (ms)",
                                   y_logscale=False,
                                   title="TTFT vs Injected Latency",
                                   per_link_delay=pld,
                                   ns=ns,
                                   subtitle_info=sub_a100_13b)
    
    fig = plot_metric_vs_injection(injection_series_13b, power_df,
                                  y_metric="output_throughput",
                                  ylabel="Per-GPU Throughput (tokens/s)",
                                  title="Throughput vs Injected Latency",
                                  subtitle_info=sub_a100_13b,
                                  per_link_delay=pld, 
                                  ns=ns,
                                  relative=True,)
    

    #%%
    metrics = ['joules_per_total_token', 'median_itl_ms', 'median_ttft_ms', 'output_throughput']
    ylabels = ['Energy per Total Token (J/token)',
               'Median ITL (ms)',
               'Median TTFT (ms)',
               'Per-GPU Throughput (tokens/s)']
    for metric, ylabel in zip(metrics, ylabels):
        fig = plot_tp2_to_tp4_extrapolation(power_df, "llama-2-13b", concurrency=32, metric=metric, 
                                            ylabel=ylabel, title="low concurrency",
                                            subtitle_info=sub_a100_8b)
        fig = plot_tp2_to_tp4_extrapolation(power_df, "llama-2-13b", concurrency=512, metric=metric, 
                                            ylabel=ylabel, title="high concurrency",
                                            subtitle_info=sub_a100_8b)
# # %%
# claude_df = pd.read_csv(f"{results_dir}latency_injection/claude_power_metrics.csv")
# injection_series_tp4_13b = build_latency_series([128,512], gpu_type='a100', 
#                                                     model_name='llama-2-13b', tp=4)

# #%%
# # 13B TP4
# fig = plot_metric_vs_injection(injection_series_tp4_13b, claude_df,
#                             y_metric="joules_per_total_token",
#                             ylabel="Energy per Total Token (J)",
#                             title="Energy Cost Per Token vs Injected Latency",
#                             subtitle_info=sub_a100_13b,
#                             relative=False)
# fig = plot_metric_vs_injection(injection_series_tp4_13b, claude_df,
#                                 y_metric="median_itl_ms",
#                                 ylabel="Inter-token Latency (ms)",
#                                 title="ITL vs Injected Latency",
#                                 subtitle_info=sub_a100_13b)
# fig = plot_metric_vs_injection(injection_series_tp4_13b, claude_df,
#                                 y_metric="median_ttft_ms",
#                                 ylabel="TTFT (ms)",
#                                 y_logscale=True,
#                                 title="TTFT vs Injected Latency",
#                                 subtitle_info=sub_a100_13b)
# fig = plot_metric_vs_injection(injection_series_tp4_13b, claude_df,
#                                 y_metric="output_throughput",
#                                 ylabel="Output Throughput (tokens/s)",
#                                 title="Throughput vs Injected Latency",
#                                 subtitle_info=sub_a100_13b,
#                                 relative=relative)

# %%
