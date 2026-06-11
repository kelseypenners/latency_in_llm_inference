#%%
import pandas as pd
import argparse
from pathlib import Path


def compute_derived_metrics(df):
    """ compute derived communication metrics by comparing multi-GPU configs to TP=1 baseline """

    # add normalized concurrency
    df['concurrency_per_gpu'] = df['max_concurrency'] / df['tp']

    # tp=1 baseline
    tp1_baseline = (
        df[df['tp'] == 1]
        .groupby(['gpu_type', 'model_name', 'concurrency_per_gpu'])
        .agg({
            'mean_itl_ms': 'mean',
            'output_throughput': 'mean',
            'mean_ttft_ms': 'mean',
            'mean_e2el_ms': 'mean',
            'p99_itl_ms': 'mean',
            'p90_itl_ms': 'mean',
        })
        .rename(columns={
            'mean_itl_ms': 'itl_tp1',
            'output_throughput': 'throughput_tp1',
            'mean_ttft_ms': 'ttft_tp1',
            'mean_e2el_ms': 'e2el_tp1',
            'p99_itl_ms': 'p99_itl_tp1',
            'p90_itl_ms': 'p90_itl_tp1',
        })
        .reset_index()
    )

    # default interconnect baseline
    default_baseline = (
        df[df['interconnect'] == 'default']
        .groupby(['gpu_type', 'model_name', 'tp', 'concurrency_per_gpu'])
        .agg({
            'mean_itl_ms': 'mean',
            'output_throughput': 'mean',
            'mean_ttft_ms': 'mean',
            'mean_e2el_ms': 'mean',
            'p99_itl_ms': 'mean',
            'p90_itl_ms': 'mean',
        })
        .rename(columns={
            'mean_itl_ms': 'itl_default',
            'output_throughput': 'throughput_default',
            'mean_ttft_ms': 'ttft_default',
            'mean_e2el_ms': 'e2el_default',
            'p99_itl_ms': 'p99_itl_default',
            'p90_itl_ms': 'p90_itl_default',
        })
        .reset_index()
    )
    # load multi-gpu data
    configs = df[df['tp'] > 1].copy()

    merged = configs.merge(
        tp1_baseline,
        on=['concurrency_per_gpu', 'model_name', 'gpu_type'],
        how='inner'
    )

    merged = merged.merge(
        default_baseline,
        on=['concurrency_per_gpu', 'model_name', 'tp', 'gpu_type'],
        how='left'
    )

    # tp=1 baseline derived metrics
    merged['itl_delta'] = merged['mean_itl_ms'] - merged['itl_tp1']
    merged['throughput_efficiency'] = (
        merged['output_throughput'] / 
        (merged['tp'] * merged['throughput_tp1'])
    )
    merged['itl_slowdown'] = merged['mean_itl_ms'] / merged['itl_tp1']
    merged['ttft_slowdown'] = merged['mean_ttft_ms'] / merged['ttft_tp1']
    merged['e2el_slowdown'] = merged['mean_e2el_ms'] / merged['e2el_tp1']

    # tail latency
    merged['p99_itl_slowdown'] = merged['p99_itl_ms'] / merged['p99_itl_tp1']
    merged['p90_itl_slowdown'] = merged['p90_itl_ms'] / merged['p90_itl_tp1']

    merged['itl_tail_ratio_p99'] = merged['p99_itl_ms'] / merged ['mean_itl_ms']
    merged['itl_tail_ratio_p90'] = merged['p90_itl_ms'] / merged ['mean_itl_ms']

    # interconnect degradation relative to default
    merged['itl_interconnect_degradation'] = merged['mean_itl_ms'] / merged['itl_default']
    merged['ttft_interconnect_degradation'] = merged['mean_ttft_ms'] / merged['ttft_default']
    merged['e2el_interconnect_degradation'] = merged['mean_e2el_ms'] / merged['e2el_default']
    merged['throughput_interconnect_efficiency'] = merged['output_throughput'] / merged['throughput_default']
    merged['p99_itl_interconnect_degradation'] = merged['p99_itl_ms'] / merged['p99_itl_default']
    merged['p90_itl_interconnect_degradation'] = merged['p90_itl_ms'] / merged['p90_itl_default']

    return merged

def main(args):

    csv_path = Path(args.experiment_csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"results csv not found: {csv_path}")
    
    df = pd.read_csv(csv_path)

    derived = compute_derived_metrics(df)
    
    experiment_dir = csv_path.parent
    out_path = experiment_dir / 'derived_metrics.csv'
    derived.to_csv(out_path, index=False)
    print(f"saved derived_metrics.csv ({len(derived)} rows)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment_csv",
        type=str,
        help="path to experiment csv"
    )

    args = parser.parse_args()

    main(args)
