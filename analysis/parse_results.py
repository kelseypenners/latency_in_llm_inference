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

def parse_sweep_summary(filepath):
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

def get_isl_osl_batch(filepath):
    # parse ISL and OSL from filename
    parts = Path(filepath).stem.split('_')
    isl, osl, batch = None, None, None
    for part in parts:
        if part.startswith('isl'):
            isl = int(part.replace('isl', ''))
        elif part.startswith('osl'):
            osl = int(part.replace('osl', ''))
        elif part.startswith('batch'):
            batch = int(part.replace('batch', ''))
    return isl, osl, batch

def get_results_bench_outputs(results_dir):
    """ get results from a dir with bench serve, latency and throughput output files """
    results_dir = Path(results_dir)
    gpu_name = results_dir.name
    
    # bench serve results
    serve_results = []
    for file in results_dir.glob('run_*_serve.txt'):
        isl, osl, _ = get_isl_osl_batch(file.name)
        metrics = parse_serve_output(file)
        metrics.update({'GPU': gpu_name, 'ISL': isl, 'OSL': osl})
        serve_results.append(metrics)

    # bench latency results
    batch_configs = {}
    for file in results_dir.glob('run_*_latency.json'):
        isl, osl, batch = get_isl_osl_batch(file.name)
        key = (isl, osl, batch)
        if key not in batch_configs:
            batch_configs[key] = {'GPU': gpu_name, 'ISL': isl, 'OSL': osl, 'batch': batch}
        batch_configs[key].update(parse_latency_json(file))

    # bench throughput results
    for file in results_dir.glob('run_*_throughput.json'):
        isl, osl, batch = get_isl_osl_batch(file.name)
        key = (isl, osl, batch)
        if key not in batch_configs:
            batch_configs[key] = {'GPU': gpu_name, 'ISL': isl, 'OSL': osl, 'batch': batch}
        batch_configs[key].update(parse_throughput_json(file))

    return serve_results, list(batch_configs.values())

def save_results_bench_outputs(gpu_dirs):

    # all results
    all_serve = []
    all_latency_throughput = []

    # gather results from both gpus
    for gpu_dir in gpu_dirs:
        if Path(gpu_dir).exists():
            # get results
            serve, latency_throughput = get_results_bench_outputs(gpu_dir)
            # add to all results
            all_serve.extend(serve)
            all_latency_throughput.extend(latency_throughput )
            print(f"{Path(gpu_dir).name}: {len(serve)} serve configs, {len(latency_throughput )} batch configs")
    
    # create dataframes with all results
    df_serve = pd.DataFrame(all_serve).sort_values(['GPU', 'ISL', 'OSL'])
    df_batch = pd.DataFrame(all_latency_throughput).sort_values(['GPU', 'ISL', 'OSL', 'batch'])
    
    # save to csvs
    df_serve.to_csv('./results/serve_results.csv', index=False)
    df_batch.to_csv('./results/latency_throughput_results.csv', index=False)
    
    print(f"\nsaved serve_results.csv ({len(df_serve)} total configs)")
    print(f"saved batch_results.csv ({len(df_batch)} total configs)")
   
def get_results_sweep(results_dir):
    """ get results from a dir with summary.json files """
    results_dir = Path(results_dir)
    gpu_type = results_dir.name

    all_results = []
    # parse all summary files in result dir
    for file in results_dir.rglob('summary*.json'):
        result = parse_sweep_summary(file)
        for run in result:
            # add gpu_type to run summary
            run["gpu_type"] = gpu_type
        all_results.extend(result)

    all_results_df = pd.DataFrame(all_results)

    # sort dataframe by gpu_type, input, and output lengths
    possible_sort_cols = ["gpu_type", "random_input_len", "random_output_len", "max_concurrency"]
    sort_cols = [c for c in possible_sort_cols if c in all_results_df.columns and all_results_df[c].notna().any()]
    all_results_df = all_results_df.sort_values(sort_cols)
    
    # move gpu_type column to front
    col = all_results_df.pop('gpu_type')
    all_results_df.insert(0, 'gpu_type', col)

    return all_results_df

# fields to keep from sweep json files
KEEP_FIELDS = ["random_input_len", "random_output_len", 
               "run_number", "max_concurrency",
               "mean_ttft_ms", "std_ttft_ms", "p99_ttft_ms", "p90_ttft_ms", "p75_ttft_ms", "mean_itl_ms", 
               "std_itl_ms", "p99_itl_ms", "p90_itl_ms", "p75_itl_ms", "mean_tpot_ms", "std_tpot_ms", 
               "p99_tpot_ms", "p90_tpot_ms", "p75_tpot_ms","output_throughput", "total_token_throughput", 
               "request_throughput", "mean_e2el_ms", "std_e2el_ms", 
               "p99_e2el_ms", "p90_e2el_ms", "p75_e2el_ms", "duration", "max_concurrent_requests",
               "completed", "failed"] 

def average_sweep_results(sweep_df):

    """ average sweep results across runs of same configs """
    df = sweep_df.copy()

    # columns we don't average over
    possible_group_cols = ['gpu_type', 'random_input_len', 'random_output_len', 'max_concurrency']
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
    
def save_results_sweep(gpu_dirs):
    """ parses and saves sweep data as csv, also saves averaged across runs csv"""
    output_dir = Path(gpu_dirs[0]).parent
    all_gpu_dfs = []

    for gpu_dir in gpu_dirs:
        gpu_result_df = get_results_sweep(gpu_dir)
        all_gpu_dfs.append(gpu_result_df)

    sweep = pd.concat(all_gpu_dfs).sort_values(["gpu_type", 
                "random_input_len", "random_output_len", "max_concurrency"])
    averaged_sweep = average_sweep_results(sweep)

    sweep.to_csv(output_dir / "sweep_results.csv", index=False)
    averaged_sweep.to_csv(output_dir / "averaged_sweep_results.csv", index=False)

    print(f"\nsaved sweep_results.csv ({len(sweep)} total runs)")
    print(f"saved averaged_sweep_results.csv ({len(averaged_sweep)} total configs)\n")

if __name__ == "__main__":

    save_results_sweep(['./results/single-gpu/27-02-2026/a100'])
