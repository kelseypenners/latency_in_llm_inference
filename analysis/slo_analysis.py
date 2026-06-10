#%%
import numpy as np
import pandas as pd

from utils.shared_config import *
from utils.plotting import *

set_figure_style()

SLOS = [
        # metric, statistic, threshold, label for csv
        ("ttft", "median", 500, "ttft_responsive_P50"),
        ("ttft", "p95", 500, "ttft_responsive_P95"),
        ("ttft", "p95", 3000, "ttft_acceptable_P95"),
        ("itl", "median", 50, "itl_tight_P50"),
        ("itl", "median", 150, "itl_responsive_P50"),
        ("itl", "p95", 150, "itl_responsive_P95"),
        ("itl", "p95", 300, "itl_acceptable_P95"),
    ]

def find_slo_capacity(df, metric_col, threshold, concurrency_col='max_concurrency'):
    """ find max concurrency under SLO attainment (interpolate if necessary) """
    if df.empty:
        return np.nan
    df = df.sort_values(concurrency_col)
    metric_values = df[metric_col].values
    concurrencies = df[concurrency_col].values

    # find point where SLOs are still met
    attainment_point = np.where(metric_values <= threshold)[0]
    if len(attainment_point) == 0:
        return 
    
    last_attaining_sample = attainment_point[-1]
    if last_attaining_sample == len(metric_values) - 1:
        return int(concurrencies[last_attaining_sample])
    
    # linearly interpolate based on last and next sample
    x0, x1 = concurrencies[last_attaining_sample], concurrencies[last_attaining_sample + 1]
    y0, y1 = metric_values[last_attaining_sample], metric_values[last_attaining_sample + 1]

    if y1 == y0:
        return int(x0)
    capacity = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0) 
    
    return int(capacity)

def build_slo_attainment_csv(df, output_path="slo_attainment.csv"):
    """ build dataframe with concurrency capacity for SLO thresholds """

    group_cols = ['gpu_type', 'model_name', 'tp', 'interconnect']
    results = []

    # go through every SLO for each config defined by group cols
    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys)) 
        for metric, stat, threshold, label in SLOS:
            col_name = f"{stat}_{metric}_ms"
            capacity = find_slo_capacity(group, col_name, threshold)
            row[label] = capacity
        
        # find capacity based on both SLOs
        row['capacity_responsive'] = min(
            row.get('ttft_responsive_P95', 0) or 0, 
            row.get('itl_responsive_P95', 0) or 0)
        row['capacity acceptable'] = min(
            row.get('ttft_acceptable_P95', 0) or 0, 
            row.get('itl_acceptable_P95', 0) or 0)
        results.append(row)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_path, index=False)
    print(f"saved SLO capacity table to {output_path}")
    return summary_df

#%%
if __name__ == "__main__":
    pass
    #%%
    results_dir = "../results/"

    df = pd.read_csv(f"{results_dir}concurrency-sweep/averaged_concurrency_sweep_results.csv")

    # a100_8b_default = build_tp_series('a100', 'llama-3.1-8b')
    # a100_8b_p2pdisabled = build_tp_series('a100', 'llama-3.1-8b', 'p2p_disabled')

    # a100_13b_default = build_tp_series('a100', 'llama-2-13b')
    # a100_13b_p2pdisabled = build_tp_series('a100', 'llama-2-13b', 'p2p_disabled')

    # fig = plot_metric_vs_concurrency(
    #     a100_13b_p2pdisabled, df,
    #     metric="median_itl_ms",
    #     ylabel="ITL",
    #     title="ITL Across TP Configs",
    #     log_scale=False,
    #     y_lim=[0, 500]
    # )

    slo_summary = build_slo_attainment_csv(df, output_path=f"{results_dir}concurrency-sweep/slo_attainment.csv")

# summary_df = pd.DataFrame(results)
# print(summary_df.to_string(index=False))
# pivot = summary_df.pivot(index="tp", columns="gpu", values="capacity")

# sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
# plt.title("SLO-Constrained Max Throughput")
# plt.ylabel("Tensor Parallelism")
# plt.xlabel("GPU Type")

# %%
