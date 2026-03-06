import json
import pandas as pd
from pathlib import Path

def parse_serve_output(filepath):
    """ parse serve output text files """
    # read file
    with open(filepath, 'r') as f:
        content = f.read()
    
    metrics= {}

    # check each line for content
    for line in content.split('\n'):
        if 'Mean TTFT (ms)' in line:
            metrics['TTFT mean'] = float(line.split()[-1])
        elif 'Median TTFT (ms)' in line:
            metrics['TTFT median'] = float(line.split()[-1])
        elif 'P99 TTFT (ms)' in line:
            metrics['TTFT P99'] = float(line.split()[-1])
        elif 'Mean TPOT (ms)' in line:
            metrics['TPOT mean'] = float(line.split()[-1])
        elif 'Median TPOT (ms)' in line:
            metrics['TPOT median'] = float(line.split()[-1])
        elif 'P99 TPOT (ms)' in line:
            metrics['TPOT P99'] = float(line.split()[-1])
        elif 'Mean ITL (ms)' in line:
            metrics['ITL mean'] = float(line.split()[-1])
        elif 'Median ITL (ms)' in line:
            metrics['ITL median'] = float(line.split()[-1])
        elif 'P99 ITL (ms)' in line:
            metrics['ITL P99'] = float(line.split()[-1])
        elif 'Request throughput (req/s)' in line:
            metrics['request throughput'] = float(line.split()[-1])
        elif 'Output token throughput (tok/s)' in line:
            metrics['output token throughput'] = float(line.split()[-1])
        elif 'Benchmark duration (s)' in line:
            metrics['duration'] = float(line.split()[-1])
        
    return metrics

def parse_latency_json(filepath):
    """ parse bench latency json files """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        'e2e latency mean': data['avg_latency'],
        'e2e latency p50': data['percentiles']['50'],
        'e2e latency p90': data['percentiles']['90'],
        'e2e latency p99': data['percentiles']['99']
    }

def parse_throughput_json(filepath):
    """ parse bench throughput json files """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return {
        'elapsed time': data['elapsed_time'],
        'requests per s': data['requests_per_second'],
        'tokens per s': data['tokens_per_second']
    }

def parse_itl_distributions(filepath):
    """ parse summary json files for itl distributions """

    with open(filepath, 'r') as f:
        summary_data = json.load(f)

    results = []
    for run in summary_data:
        if 'itls' not in run:
            continue
        itls_flat = [itl * 1000 for itl in sum(run['itls'], [])] 
        results.append({
            'max_concurrency': run['max_concurrency'],
            'run_number': run['run_number'],
            'itls_ms': itls_flat
        })
    return results

def get_itl_distributions(results_dir):
    """ get itl distributions from a dir with summary.json files """
    results_dir = Path(results_dir)
    gpu_type = results_dir.name

    all_results = []
    # parse all summary files in results dir
    for file in results_dir.rglob('summary*.json'):
        run_results = parse_itl_distributions(file)
        for run in run_results:
            run['gpu_type'] = gpu_type
        all_results.extend(run_results)

    df = pd.DataFrame(all_results)
    df = df.sort_values(['gpu_type', 'max_concurrency', 'run_number'])
    col = df.pop('gpu_type')
    df.insert(0, 'gpu_type', col)
    return df

def save_itl_distributions(gpu_dirs, output_dir):
    """ parses and saves itl data as parquet """

    output_dir = Path(output_dir)
    all_gpu_dfs = []

    for gpu_dir in gpu_dirs:
        gpu_df = get_itl_distributions(gpu_dir)
        all_gpu_dfs.append(gpu_df)
        print(f"{Path(gpu_dir).name}: {len(gpu_df)} runs collected")

    df = pd.concat(all_gpu_dfs).sort_values(['gpu_type', 'max_concurrency', 'run_number'])
    df.to_parquet(output_dir / 'itl_distributions.parquet', index=False)
    print(f"\nsaved itl_distributions.parquet ({len(df)} total runs)")

# fields to keep from sweep json files
KEEP_FIELDS = ["random_input_len", "random_output_len", 
               "run_number", "max_concurrency",
               "mean_ttft_ms", "std_ttft_ms", "p99_ttft_ms", "p90_ttft_ms", "p75_ttft_ms", "mean_itl_ms", 
               "std_itl_ms", "p99_itl_ms", "p90_itl_ms", "p75_itl_ms", "mean_tpot_ms", "std_tpot_ms", 
               "p99_tpot_ms", "p90_tpot_ms", "p75_tpot_ms","output_throughput", "total_token_throughput", 
               "request_throughput", "mean_e2el_ms", "std_e2el_ms", 
               "p99_e2el_ms", "p90_e2el_ms", "p75_e2el_ms", "duration", "max_concurrent_requests",
               "completed", "failed"] 

def parse_sweep_summary_json(filepath):
    """ parse sweep summary json files """
    
    with open(filepath, 'r') as f:
        summary_data = json.load(f)
    results = []

    # for each run get kept field values
    for run in summary_data:
        run_results={}
        for k in KEEP_FIELDS:
            keep_value = run.get(k)
            run_results[k] = keep_value
        results.append(run_results)
    return results

def get_results_sweep(results_dir):
    """ get results from a dir with summary.json files """
    results_dir = Path(results_dir)
    gpu_type = results_dir.name

    all_results = []
    # parse all summary files in result dir
    for file in results_dir.rglob('summary*.json'):
        result = parse_sweep_summary_json(file)
        for run in result:
            # add gpu_type to run summary
            run["gpu_type"] = gpu_type
        all_results.extend(result)

    all_results_df = pd.DataFrame(all_results)

    return all_results_df

def average_sweep_results(sweep_df):
    """ average sweep results across runs of same configs """
    df = sweep_df.copy()

    # columns we don't average over
    possible_group_cols = ['gpu_type', 'random_input_len', 'random_output_len', 'max_concurrency', 'dataset-name']
    group_cols = [c for c in possible_group_cols if c in df.columns and df[c].notna().any()]
    drop_cols = group_cols + ['run_number']

    # columns to average
    avg_cols = df.drop(columns=drop_cols).columns.tolist()
    std_cols = ['mean_ttft_ms', 'mean_itl_ms', 'mean_tpot_ms', 'output_throughput', 'total_token_throughput', 'mean_e2el_ms', 'duration']

    # average for each config group 
    averaged_sweep = sweep_df.groupby(group_cols)[avg_cols].mean().reset_index()

    # add run counts to df
    run_counts = sweep_df.groupby(group_cols)['run_number'].count().reset_index()
    run_counts = run_counts.rename(columns={'run_number': 'num_runs'})
    averaged_sweep = averaged_sweep.merge(run_counts, on=group_cols)

    # add std across runs to df
    std_across_runs = sweep_df.groupby(group_cols)[std_cols].std().reset_index()
    std_across_runs = std_across_runs.rename(columns={c: f'std_runs_{c}' for c in std_cols})

    averaged_sweep = averaged_sweep.merge(std_across_runs, on=group_cols)

    return averaged_sweep
    
def save_results_sweep(gpu_dirs, output_dir, sort_cols):
    """ parses and saves sweep data as csv, also saves averaged across runs csv"""

    all_gpu_dfs = []
    for gpu_dir in gpu_dirs:
        gpu_result_df = get_results_sweep(gpu_dir)
        all_gpu_dfs.append(gpu_result_df)

    sweep = pd.concat(all_gpu_dfs).sort_values(sort_cols)
    averaged_sweep = average_sweep_results(sweep)

    sweep.to_csv(output_dir / f"{output_dir.name.replace('-', '_')}_results.csv", index=False)
    averaged_sweep.to_csv(output_dir / f"averaged_{output_dir.name.replace('-', '_')}_results.csv", index=False)

    print(f"saved \n{output_dir.name.replace('-', '_')}_results.csv ({len(sweep)} total runs)")
    print(f"saved averaged_{output_dir.name.replace('-', '_')}_results.csv({len(averaged_sweep)} total configs)\n")
    
if __name__ == "__main__":
    base = Path('./results/single-gpu')
    # save_results_sweep(
    #     gpu_dirs=[base / 'seqlen-sweep/a100', base / 'seqlen-sweep/v100'],
    #     output_dir=base / 'seqlen-sweep',
    #     sort_cols=['gpu_type', 'random_input_len', 'random_output_len']
    # )

    # save_results_sweep(
    #     gpu_dirs=[base / 'concurrency-sweep/a100'],
    #     output_dir=base / 'concurrency-sweep',
    #     sort_cols=['gpu_type', 'max_concurrency']
    # )

    save_itl_distributions(
        gpu_dirs=[base / 'concurrency-sweep/a100'],
        output_dir=base / 'concurrency-sweep',
    )