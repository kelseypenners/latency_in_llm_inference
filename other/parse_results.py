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



def get_results(results_dir):
    results_dir = Path(results_dir)
    gpu_name = results_dir.name  # 'v100' or 'a100'
    
    # SERVE RESULTS
    serve_results = []
    for file in results_dir.glob('run_*_serve.txt'):
        isl, osl, _ = get_isl_osl_batch(file.name)
        metrics = parse_serve_output(file)
        metrics.update({'GPU': gpu_name, 'ISL': isl, 'OSL': osl})
        serve_results.append(metrics)
    
    # BATCH RESULTS
    batch_configs = {}
    for file in results_dir.glob('run_*_latency.json'):
        isl, osl, batch = get_isl_osl_batch(file.name)
        key = (isl, osl, batch)
        if key not in batch_configs:
            batch_configs[key] = {'GPU': gpu_name, 'ISL': isl, 'OSL': osl, 'batch': batch}
        batch_configs[key].update(parse_latency_json(file))

    for file in results_dir.glob('run_*_throughput.json'):
        isl, osl, batch = get_isl_osl_batch(file.name)
        key = (isl, osl, batch)
        if key not in batch_configs:
            batch_configs[key] = {'GPU': gpu_name, 'ISL': isl, 'OSL': osl, 'batch': batch}
        batch_configs[key].update(parse_throughput_json(file))
    
    return serve_results, list(batch_configs.values())

def get_results(results_dir):
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
   
if __name__ == "__main__":

    all_serve = []
    all_latency_throughput = []

    # Collect from both GPUs
    for gpu_dir in ['./results/single-gpu/v100/', './results/single-gpu/a100/']:
        if Path(gpu_dir).exists():
            serve, latency_throughput = get_results(gpu_dir)
            all_serve.extend(serve)
            all_latency_throughput.extend(latency_throughput )
            print(f"{Path(gpu_dir).name}: {len(serve)} serve configs, {len(latency_throughput )} batch configs")
    
    # Create combined DataFrames
    df_serve = pd.DataFrame(all_serve).sort_values(['GPU', 'ISL', 'OSL'])
    df_batch = pd.DataFrame(all_latency_throughput).sort_values(['GPU', 'ISL', 'OSL', 'batch'])
    
    # Save to single CSVs
    df_serve.to_csv('./results/serve_results.csv', index=False)
    df_batch.to_csv('./results/latency_throughput_results.csv', index=False)
    
    print(f"\nsaved serve_results.csv ({len(df_serve)} total configs)")
    print(f"saved batch_results.csv ({len(df_batch)} total configs)")
