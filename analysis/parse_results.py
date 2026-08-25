import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def metadata_from_path(path: Path):
    """ infer metadata (gpu_type, tp, etc) from path of a summary file """

    parts = path.parts

    experiment = None
    gpu_type = None
    tp = None
    interconnect = None

    for i, part in enumerate(parts):
        if part in ('a100', 'v100'):
            gpu_type = part
            # find experiment & model name from path
            if i >= 2:
                experiment = parts[i-2]
                model_name = parts[i-1]
            # expect tp after gpu_type
            if i + 1 < len(parts) and parts[i+1].startswith('tp'):
                try:
                    tp = int(parts[i+1][2:])
                except ValueError:
                    pass
            # expect interconnect after tp
            if i + 2 < len(parts):
                interconnect = parts[i+2]
            break
    return {
        'experiment':   experiment,
        'model_name':   model_name,
        'gpu_type':     gpu_type,
        'tp':           tp,
        'interconnect': interconnect,
    }

def parse_sweep_summary_json(filepath: Path):
    """ parse summary json files """

    # load json
    with open(filepath, 'r') as f:
        summary_data = json.load(f)
    
    # ignore keys with large amounts of data (processed separately)
    drop_keys = {
        "input_lens",
        "output_lens",
        "ttfts",
        "itls",
        "generated_texts",
        "errors",
    }

    # ignore drop keys before dataframe creation
    filtered_data = []
    for run in summary_data:
        filtered_run = {}
        for k,v in run.items():
            if k not in drop_keys:
                filtered_run[k] = v
        filtered_data.append(filtered_run)

    df = pd.DataFrame(filtered_data)

    # add metadata from path
    metadata = metadata_from_path(filepath)
    for k, v in metadata.items():
        df[k] = v

    return df

def build_summary_csvs(results_dir: Path):
    """ generate missing summary.csvs from json files """

    # find every summary.json
    for summary_json in results_dir.rglob("summary.json"):

        parent_dir = summary_json.parent
        timestamp_dir = parent_dir.parent
        summary_csv = timestamp_dir / "summary.csv"

        # skip if already exists
        if summary_csv.exists():
            print(f"    skipping existing: {summary_csv}")
            continue

        print(f"generating: {summary_csv}")

        # find all bench directories
        benchmark_dirs = []
        for dir in timestamp_dir.iterdir():
            if dir.is_dir() and dir.name.startswith(("BENCH", "SERVE")):
                benchmark_dirs.append(dir)

        # collect all dfs from bench directories in timestamp dir
        all_dfs = []
        for bench_dir in benchmark_dirs:
            bench_summary = bench_dir / "summary.json"

            if not bench_summary.exists():
                continue

            try:
                df = parse_sweep_summary_json(bench_summary)
                all_dfs.append(df)
            except Exception as e:
                print(f"failed parsing {bench_summary}: {e}")

        if not all_dfs:
            print(f"no summaries found in {parent_dir.parent}")
            continue

        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(summary_csv)

        print(f"saved: {summary_csv}")

def load_results(experiment_dir: Path):
    """ load all csvs into one aggregate csv for given experiment dir """

    all_dfs = []
    for summary_csv in experiment_dir.rglob("summary.csv"):
        try:
            df = pd.read_csv(summary_csv)
            metadata = metadata_from_path(summary_csv)
            for k, v in metadata.items():
                df[k] = v
            # config path to store gpu ids and latency injection values
            config_path = summary_csv.parent.parent / "config_merged.json"

            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)

                gpu_ids = config.get("gpu_ids")
                injected_latency = config.get("injected_latency_us")

                # add gpu_ids
                if gpu_ids is not None:
                    df["gpu_ids"] = gpu_ids

                # add injected latency
                if injected_latency is not None:
                    df["injected_latency_us"] = injected_latency

            all_dfs.append(df)
        except Exception as e:
            print(f"  skipping {summary_csv}: {e}")

    if not all_dfs:
        print(f"no CSVs found in {experiment_dir}")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    if 'Unnamed: 0' in combined.columns:
        combined = combined.drop(columns=['Unnamed: 0'])

    output_path = experiment_dir / (experiment_dir.name.replace('-','_') + '_results.csv')
    combined.to_csv(output_path, index=False)
    print(f"saved {output_path} ({len(combined)} rows)")
    return combined


def average_results(results_csv_path: Path):
    """ average results across runs of identical configs """

    df = pd.read_csv(results_csv_path)

    # columns that identify a configuration
    possible_group_cols = [
        'max_concurrency',
        'model_name', 
        'backend', 
        'endpoint_type', 
        'gpu_type',
        'tp',
        'interconnect',
        'random-input-len',
        'random-output-len',
        'random_input_len',
        'random_output_len',
        'sharegpt-output-len',
        'enable-prefix-caching',
        'dataset-name',
        'dataset-path',
        'injected_latency_us',
    ]

    # only group by existing group columns
    group_cols = [c for c in possible_group_cols if c in df.columns]
    drop_cols = group_cols + ['run_number']

    # select columns to average over
    metric_cols = df.columns.difference(drop_cols)
    metric_cols = df[metric_cols].select_dtypes(include="number").columns

    std_metrics = [
        'duration',
        'mean_e2el_ms', 'median_e2el_ms',
        'mean_itl_ms', 'median_itl_ms',
        'mean_tpot_ms', 'median_tpot_ms',
        'mean_ttft_ms', 'median_ttft_ms',
        'output_throughput', 'request_throughput'
    ]
    std_metric_cols = [col for col in std_metrics if col in metric_cols]

    # group data
    grouped = df.groupby(group_cols, dropna=False)
    
    # calculate mean and standard deviation
    means = grouped[metric_cols].mean()
    stds = grouped[std_metric_cols].std()

    # calculate percent standard deviation
    pct_std = (stds / means[std_metric_cols].abs() * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    # rename columns to std_runs_{metric}
    pct_std.columns = [f"std_runs_{col}" for col in std_metric_cols]

    # combine means and percent standard deviations
    averaged = pd.concat([means, pct_std], axis=1).reset_index()
    
    output_path = results_csv_path.parent / ('averaged_' + results_csv_path.name)
    averaged.to_csv(output_path, index=False)
    print(f"saved {output_path} ({len(averaged)} rows)")

    return averaged


def main(args):

    exp_dir = Path(args.experiment_dir)

    if not exp_dir.exists():
        raise FileNotFoundError(f"experiment not found: {exp_dir}")

    print(f"processing experiment: {exp_dir}")

    # build missing summary csvs (from resumed runs)
    build_summary_csvs(exp_dir)

    df = load_results(exp_dir)

    results_csv = exp_dir / (exp_dir.name.replace("-", "_") + "_results.csv")
    if not results_csv.exists():
        print("no results CSV found yet...")
        return

    average_results(results_csv)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment_dir",
        type=str,
        help="path to experiment folder"
    )

    args = parser.parse_args()

    main(args)





