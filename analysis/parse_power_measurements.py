import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import argparse

def get_gpu_ids(gpu_ids_val):
    """ parse gpu_ids from results csv into list of ints """
    if pd.isna(gpu_ids_val):
        return None
    gpu_ids = []
    for x in str(gpu_ids_val).split(","):
        gpu_ids.append(int(x.strip()))
    return gpu_ids

def parse_run_power(run_row, power_df):
    """ calculates power metrics for a single benchmark run """

    power_metrics = {
        'total_energy_joules': None,
        'joules_per_output_token': None,
        'joules_per_total_token': None,
        'mean_gpu_utilization': None,
        'num_power_samples': None,
    }

    if pd.isna(run_row.get('date')) or pd.isna(run_row.get('duration')) or power_df is None:
        return pd.Series(power_metrics)

    end_time = datetime.strptime(str(run_row['date']).strip(), "%Y%m%d-%H%M%S")
    duration = float(run_row['duration'])
    start_time = end_time - timedelta(seconds=duration)

    # get gpu ids for power log  parsing
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
        return pd.Series(power_metrics)
    
    # convert string timestamps to compatible datetime objects
    power_df = power_df.copy()
    power_df['timestamp'] = pd.to_datetime(power_df['timestamp'].str.strip())

    # filter power log to run window and active GPUs
    mask = (
        (power_df['timestamp'] >= start_time) & 
        (power_df['timestamp'] <= end_time)& 
        power_df['index'].isin(gpu_ids)
    )
    window_df = power_df[mask].copy()

    if window_df.empty:
        print(f"    warning: no power samples found for run window "
              f"{start_time} -> {end_time}, GPUs {gpu_ids}")
        return pd.Series(power_metrics)
    
    total_energy = 0.0
    window_df = window_df.sort_values(by=['index', 'timestamp'])
    num_samples = len(window_df)

    # calculate total energy used for each gpu
    for gpu_id in gpu_ids:

        gpu_data = window_df[window_df['index'] == gpu_id]
        timestamps = gpu_data['timestamp'].values
        powers = gpu_data['power.draw'].values
        dt_seconds = (timestamps - timestamps[0]) / np.timedelta64(1, 's')

        # trapezoidal integration
        dt = np.diff(dt_seconds)
        avg_powers = (powers[:-1] + powers[1:]) / 2.0
        total_energy += np.sum(avg_powers * dt)

    output_tokens = float(run_row.get('total_output_tokens', 0))
    input_tokens = float(run_row.get('total_input_tokens', 0))
    total_tokens = output_tokens + input_tokens
    
    power_metrics['total_energy_joules'] = total_energy
    power_metrics['joules_per_output_token'] = total_energy / output_tokens if output_tokens > 0 else None
    power_metrics['joules_per_total_token'] = total_energy / total_tokens if total_tokens > 0 else None
    power_metrics['mean_gpu_utilization'] = window_df['utilization.gpu'].mean()
    power_metrics['num_power_samples'] = num_samples

    return pd.Series(power_metrics)

def find_run_power(run_row, cached_dirs):
    """ returns matching power log data based on row's first timestamp """

    if pd.isna(run_row.get('date')):
        return None

    # expected columns for power log
    EXPECTED_COLUMNS = ['timestamp', 'index', 'power.draw', 'utilization.gpu', 
                        'utilization.memory', 'memory.used', 'memory.total'] 

    # parse benchmark run for datetime
    try:
        run_date_str = str(run_row['date']).strip()
        run_dt = datetime.strptime(run_date_str,  "%Y%m%d-%H%M%S") 
    except ValueError:
        print(f" error parsing date: {run_row.get('date')}")
        return None

    best_match_path = None
    smallest_diff = timedelta(days=2)

    # find matching power log based on run dir timestamp
    for curr_dir_dt, curr_run_dir in cached_dirs:
        if curr_dir_dt <= run_dt:

            time_diff = run_dt - curr_dir_dt
            if time_diff < smallest_diff:
                smallest_diff = time_diff
                best_match_path = curr_run_dir
        else:
            # if curr_dir_dt > run_dt, rest of sorted cached dirs will be in future 
            break


    if best_match_path is None:
        # print(f"no matching dir found within results directory with timstamp {run_date_str}")
        return None
    print(f"         match!!{best_match_path} , {run_dt}")
    power_log_path = best_match_path / "gpu_power.csv"

    if power_log_path.exists():
        with open(power_log_path, 'r') as f:
            first_line = f.readline().lower()
        if 'timestamp' in first_line or 'index' in first_line:
            power_log_data = pd.read_csv(power_log_path)
        else:
            # if no headers, add them manually
            print(f"detected no headers, adding manually {power_log_path}")
            power_log_data = pd.read_csv(power_log_path, names=EXPECTED_COLUMNS)
            power_log_data.to_csv(power_log_path, index=False)
    else: 
        print(f"dir found({best_match_path.name}), but no gpu_power.csv found")
        return None

    return power_log_data

def build_power_csv(experiment_dir: Path, results_csv_path: Path):
    
    results_df = pd.read_csv(results_csv_path)

    # cache dirs and their timestamps
    cached_dirs = []
    for curr_run_dir in experiment_dir.rglob("run_*"):
        try:
            curr_date_str = "-".join(curr_run_dir.name.split('_')[-2:])
            curr_dir_dt = datetime.strptime(curr_date_str,  "%Y%m%d-%H%M%S")
            cached_dirs.append((curr_dir_dt, curr_run_dir))
        except (ValueError, IndexError):
            continue

    cached_dirs.sort(key=lambda x: x[0])
    power_rows = []

    # get power data for each row
    for _, row in results_df.iterrows():

        power_log_data = find_run_power(row, cached_dirs)
        power_rows.append(parse_run_power(row, power_log_data))
    
    # add power data to run data
    power_df = pd.DataFrame(power_rows).reset_index(drop=True)
    all_data = pd.concat([results_df, power_df], axis=1)

    # save power csv
    power_csv_path = experiment_dir / "power_metrics.csv"
    all_data.to_csv(power_csv_path, index=False)
    print(f"saved: {power_csv_path}")

    return all_data

def main(args):
    experiment_dir = Path(args.experiment_dir)
    results_csv_path = Path(args.results_csv_path)

    if not experiment_dir.exists():
        raise FileNotFoundError(f"experiment dir not found: {experiment_dir}")

    if not results_csv_path.exists():
        raise FileNotFoundError(f"results CSV not found: {results_csv_path}")

    df = build_power_csv(experiment_dir, results_csv_path)
    print("done. rows:", len(df))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment_dir",
        type=str,
        help="path to experiment folder"
    )
    parser.add_argument(
        "results_csv_path",
        type=str,
        help="path to results CSV"
    )

    args = parser.parse_args()

    main(args)