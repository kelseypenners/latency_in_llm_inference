import json
import pandas as pd
from pathlib import Path

from parse_results import *



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