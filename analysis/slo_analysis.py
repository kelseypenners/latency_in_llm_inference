import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from utils.shared_config import *
from utils.plotting import *

SLOS = [
        # metric, statistic, threshold, label for csv
        ("ttft", "median", 500, "ttft_responsive_P50"),
        ("itl", "median", 150, "itl_responsive_P50"),

        ("ttft", "p95", 1000, "ttft_responsive_P95"),
        ("ttft", "p95", 5000, "ttft_acceptable_P95"),
        ("itl", "p95", 100, "itl_responsive_P95"),
        ("itl", "p95", 250, "itl_acceptable_P95"),

        ("ttft", "p90", 1000, "ttft_responsive_P90"),
        ("ttft", "p90", 5000, "ttft_acceptable_P90"),
        ("itl", "p90", 100, "itl_responsive_P90"),
        ("itl", "p90", 250, "itl_acceptable_P90")

    ]

def find_slo_limit(df, metric_col, threshold, target_col='max_concurrency'):
    """ find max target value (concurrency or latency) under SLO attainment (interpolate if necessary) """
    if df.empty:
        return np.nan
    df = df.sort_values(target_col)
    metric_values = df[metric_col].values
    target_values = df[target_col].values

    # find point where SLOs are still met
    attainment_point = np.where(metric_values <= threshold)[0]
    if len(attainment_point) == 0:
        return
    
    last_attaining_sample = attainment_point[-1]
    if last_attaining_sample == len(metric_values) - 1:
        return int(target_values[last_attaining_sample])
    
    # linearly interpolate based on last and next sample
    x0, x1 = target_values[last_attaining_sample], target_values[last_attaining_sample + 1]
    y0, y1 = metric_values[last_attaining_sample], metric_values[last_attaining_sample + 1]

    if y1 == y0:
        return int(x0)
    limit = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0) 
    
    return int(limit)


def build_slo_attainment_csv(df, target_col, output_path="slo_attainment.csv"):
    """ build dataframe with target limits for SLO thresholds """
 
    group_cols = ['gpu_type', 'model_name', 'tp', 'interconnect']
    if target_col == 'injected_latency_us':
        group_cols.append('max_concurrency')
    # if target_col == 'max_concurrency':
    #     group_cols.append('injected_latency_us')

    results = []

    # go through every SLO for each config defined by group cols
    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys)) 
        for metric, stat, threshold, label in SLOS:
            col_name = f"{stat}_{metric}_ms"
            limit = find_slo_limit(group, col_name, threshold, target_col=target_col)
            row[label] = limit
        
        # find combined limit based on both SLOs
        responsive_limit_95 = min(
            row.get('ttft_responsive_P95', 0) or 0, 
            row.get('itl_responsive_P95', 0) or 0)
        responsive_limit_90 = min(
            row.get('ttft_responsive_P90', 0) or 0, 
            row.get('itl_responsive_P90', 0) or 0)
        acceptable_limit_95 = min(
            row.get('ttft_acceptable_P95', 0) or 0, 
            row.get('itl_acceptable_P95', 0) or 0)
        acceptable_limit_90 = min(
            row.get('ttft_acceptable_P90', 0) or 0, 
            row.get('itl_acceptable_P90', 0) or 0)
        row['limit_responsive_95'] = responsive_limit_95
        row['limit_responsive_90'] = responsive_limit_90
        row['limit_acceptable_95'] = acceptable_limit_95
        row['limit_acceptable_90'] = acceptable_limit_90

        if target_col == 'max_concurrency': 
            tp = row.get('tp', 1)
            row['pergpu_limit_responsive_90'] = int(responsive_limit_90 / tp)
            row['pergpu_limit_responsive_95'] = int(responsive_limit_95 / tp)
            row['pergpu_limit_acceptable_90'] = int(acceptable_limit_90 / tp) 
            row['pergpu_limit_acceptable_95'] = int(acceptable_limit_95 / tp) 
        
        results.append(row)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_path, index=False)
    print(f"saved SLO table to {output_path}")
    return summary_df

def main(args):

    results_csv_path = Path(args.results_csv)
    if not results_csv_path.exists():
            raise FileNotFoundError(f"results csv not found {results_csv_path}")

    df = pd.read_csv(results_csv_path)

    target_col = 'injected_latency_us' if args.mode == 'latency' else 'max_concurrency'

    experiment_dir = results_csv_path.parent
    out_path = experiment_dir / 'slo_attainment.csv'
    
    build_slo_attainment_csv(df, target_col=target_col, output_path=out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "results_csv",
        type=str,
        help="path to results csv"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=['concurrency', 'latency'],
        default='concurrency',
        help="which mode to find SLO limits"
    )

    args = parser.parse_args()

    main(args)
