#%%
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from analysis.parse_power_measurements import *
from utils.plotting import *

# set_figure_style()

def plot_run_power(run_row, cached_dirs, gpu_id=None, window_pct_start=None):
    """ plot power draw over time for single run """

    # find corresponding power log
    power_df = find_run_power(run_row, cached_dirs)

    if pd.isna(run_row.get('date')) or pd.isna(run_row.get('duration')) or power_df is None:
        print("error with finding power log data")
        return

    end_time = datetime.strptime(str(run_row['date']).strip(), "%Y%m%d-%H%M%S")
    duration = float(run_row['duration'])
    start_time = end_time - timedelta(seconds=duration)

    # get gpu ids for power log parsing
    gpu_ids = get_gpu_ids(run_row.get('gpu_ids'))
    if gpu_ids is None:
        tp = int(run_row.get('tp', 1))
        gpu_type = run_row.get('gpu_type')

        # hard-coded fallback
        if gpu_type == 'a100':
            if tp == 1: gpu_ids = [7]
            elif tp == 2: gpu_ids = [6,7]
            elif tp == 4: gpu_ids = [4,5,6,7]
        elif gpu_type == 'v100':  
            if tp == 1: gpu_ids = [3]
            elif tp == 2: gpu_ids = [1,2]
            elif tp == 4: gpu_ids = [0,1,2,3]

    if not gpu_ids:
        print("error finding gpu ids")
        return

    # set window duration if thats what we want
    if window_pct_start:
        x_min = duration * window_pct_start
        x_max = x_min + 2
    
    # convert string timestams to compatible datatime objects
    power_df = power_df.copy()
    power_df['timestamp'] = pd.to_datetime(power_df['timestamp'].str.strip())

    # filter power log to run window and active GPUs
    mask = (
        (power_df['timestamp'] >= start_time) & 
        (power_df['timestamp'] <= end_time) & 
        power_df['index'].isin(gpu_ids)
    )
    window_df = power_df[mask].copy()

    if window_df.empty:
        print(f"    warning: no power samples found in run window {start_time} -> {end_time}")
        return
    

    window_df = window_df.sort_values(by=['index', 'timestamp'])

    fig, ax = plt.subplots(figsize=(6, 4))
    if gpu_id:
        gpu_ids = [gpu_id]

    for gpu_id in gpu_ids:
        gpu_data = window_df[window_df['index'] == gpu_id]
        if gpu_data.empty:
            continue
            
        seconds_from_start = (gpu_data['timestamp'] - start_time).dt.total_seconds()
        powers = gpu_data['power.draw'].values
        
        if window_pct_start:
            window_mask = (seconds_from_start >= x_min) & (seconds_from_start <= x_max)
            seconds_from_start = seconds_from_start[window_mask]
            powers = powers[window_mask]

        ax.plot(seconds_from_start, powers, marker='o', 
                markersize=0, linestyle='-', linewidth=0.5, label=f'GPU {gpu_id}')
        
    ax.set_title(f"Power Consumption Profile", pad=20)
    ax.set_xlabel("Time (seconds from run start)")
    ax.set_ylabel("Power Draw (Watts)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="upper right")

    subtitle(ax, model=run_row["model_name"],
              added_latency=fr"{run_row.get('injected_latency_us', 0):.0f} $\mu$s injected",
             max_concurrency=f"concurrency = {run_row["max_concurrency"]}",
             run_date=f"run timestamp: {run_row['date']}")
    
    plt.tight_layout()

if __name__ == "__main__":
    pass
    #%%
    experiment_dir = Path("../results/latency_injection")
    results_csv_path = Path("../results/latency_injection/latency_injection_results.csv")

    results_df = pd.read_csv(results_csv_path)

    # cache dirs and their timestamps
    cached_dirs = []
    for curr_run_dir in experiment_dir.rglob("run_*"):
        try:
            curr_date_str = "-".join(curr_run_dir.name.split('_')[-2:])
            curr_dir_dt = datetime.strptime(curr_date_str, "%Y%m%d-%H%M%S")
            cached_dirs.append((curr_dir_dt, curr_run_dir))
        except (ValueError, IndexError):
            continue
    cached_dirs.sort(key=lambda x: x[0])

    for x in [0,100,500,1000, 1500, 2000]:

        filtered_df = filter_df(results_df, model_name="llama-2-13b", tp=2, injected_latency_us=x, max_concurrency=512)

        for _, row in filtered_df.iterrows():

            # plot_run_power(row, cached_dirs, gpu_id=6)
            # plot_run_power(row, cached_dirs, gpu_id=7)
            plot_run_power(row, cached_dirs)
            plot_run_power(row, cached_dirs, window_pct_start=0.7)
    
    #%%
    #%%
    experiment_dir = Path("../results/concurrency-sweep")
    results_csv_path = Path("../results/concurrency-sweep/concurrency_sweep_results.csv")

    results_df = pd.read_csv(results_csv_path)

    # cache dirs and their timestamps
    cached_dirs = []
    for curr_run_dir in experiment_dir.rglob("run_*"):
        try:
            curr_date_str = "-".join(curr_run_dir.name.split('_')[-2:])
            curr_dir_dt = datetime.strptime(curr_date_str, "%Y%m%d-%H%M%S")
            cached_dirs.append((curr_dir_dt, curr_run_dir))
        except (ValueError, IndexError):
            continue
    cached_dirs.sort(key=lambda x: x[0])

    for x in [4,8,16,32,64,128, 256,512]:

        filtered_df = filter_df(results_df, model_name="llama-3.2-1b", tp=2,
                                 max_concurrency=x, gpu_type='a100', interconnect='default')

        for _, row in filtered_df.iterrows():

            # plot_run_power(row, cached_dirs, gpu_id=6)
            # plot_run_power(row, cached_dirs, gpu_id=7)
            plot_run_power(row, cached_dirs)
            #plot_run_power(row, cached_dirs, window_pct_start=0.7)
    
# %%
