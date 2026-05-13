import json
import pandas as pd
from pathlib import Path

from parse_results import *


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
    """ save results from initial runs """
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

def save_prompt_type(results_dir):
    """ creates prompt type column in dataset-comparison summary csv """

    df = pd.read_csv(results_dir)
    
    def classify(path):
        if 'harder_' in str(path):
            return 'hard'
        elif 'easy_' in str(path):
            return 'easy'
        else:
            return 'random'

    df['prompt_type'] = df['dataset-path'].apply(classify)
    df.to_csv(results_dir)
    print(f"saved {results_dir}")

def parse_itl_distributions(filepath, nprompts=50):
    """ parse summary json files for itl distributions """

    with open(filepath, 'r') as f:
        summary_data = json.load(f)

    results = []
    for run in summary_data:
        if 'itls' not in run:
            continue
        selected = run['itls'][::len(run['itls']) // nprompts]
        itls_flat = [itl * 1000 for itl in sum(selected, [])] 
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
    print(f"saved itl_distributions.parquet ({len(df)} total runs)\n")