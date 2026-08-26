import numpy as np
import pandas as pd
import argparse
import json

from utils.shared_config import *
from utils.plotting import *

set_figure_style()


def num_allreduce_per_step(num_layers):
    """ 2 AllReduce per transformer layer (one after attention, one after FFN) """
    return 2 * num_layers

def allreduce_message_size_bytes(batch_size, hidden_size, seq_len=1, dtype_bytes=2):
    """ size of input tensor passed to AllReduce during decode """
    # shape is [batch_size, seq_len, hidden_size]
    return batch_size * hidden_size * dtype_bytes * seq_len

def lookup_allreduce_latency(df, server, interconnect, tp, message_bytes):
    """ lookup AllReduce latency measurement based on config """
    filtered_df = filter_df(df, interconnect=interconnect, tp=tp, server=server)
    if filtered_df.empty:
        return None
    row = filtered_df.iloc[0]
    sizes = json.loads(row['nccl_message_sizes'])
    latencies = json.loads(row['nccl_latency'])

    # linear interpolation based on measurements
    latency_ms = np.interp(message_bytes, sizes, latencies)
    return latency_ms / 1000

def predicted_comm_overhead_per_step(df, server, batch_size, model, interconnect, tp=2):
    """ total AllReduce communication overhead in milliseconds for one decode step """
    message_size = allreduce_message_size_bytes(batch_size, model["hidden_size"])
    num_ops = num_allreduce_per_step(model['num_layers'])
    latency = lookup_allreduce_latency(df, server=server, interconnect=interconnect, 
                                       tp=tp, message_bytes=message_size)
    if latency is None:
        return np.nan
    return (num_ops * latency)

def merge_measurements_onto_df(df, nccl_df):
    """ add interconnect measurement data to derived dataframe """
    df_copy = df.copy()
    predicted_comm_ms = []
    single_allreduce_lat_ms = []

    model_name_to_cfg = {cfg["model_name"]: cfg for cfg in MODELS.values() if "model_name" in cfg}

    for _, row in df_copy.iterrows():
        row_model_name = row['model_name']
        
        if row_model_name not in model_name_to_cfg:
            predicted_comm_ms.append(np.nan)
            single_allreduce_lat_ms.append(np.nan)
            continue
            
        model_cfg = model_name_to_cfg[row_model_name]
        batch_size = row['max_concurrency']
        interconnect = row['interconnect']
        tp_degree = row['tp']
        server = row.get('server', 'shy-fec')
        
        # predict communication overhead based on test measurements
        comm_ms = predicted_comm_overhead_per_step(
            nccl_df, server=server, batch_size=batch_size, 
            model=model_cfg, interconnect=interconnect, tp=tp_degree
        )
        predicted_comm_ms.append(comm_ms)

        message_size = allreduce_message_size_bytes(batch_size, model_cfg["hidden_size"])
        lat_ms = lookup_allreduce_latency(
            nccl_df, server=server, interconnect=interconnect, 
            tp=tp_degree, message_bytes=message_size
        )
        single_allreduce_lat_ms.append(lat_ms)
        
    df_copy['predicted_comm_ms'] = predicted_comm_ms
    df_copy['allreduce_lat_ms'] = single_allreduce_lat_ms
    return df_copy


def add_predicted_itl_deltas(df):
    """ compute observed and predicted ITL deltas relative to default  """

    df = df.copy()
    key_cols = ['model_name', 'tp', 'max_concurrency', 'gpu_type']

    # get default row for each config
    default_df = (
        df[df['interconnect'] == 'default']
        [key_cols + ['mean_itl_ms', 'predicted_comm_ms']]
        .rename(columns={
            'mean_itl_ms': 'itl_default_ms',
            'predicted_comm_ms': 'predicted_comm_default_ms',
        })
    )

    merged = df.merge(default_df, on=key_cols, how='left')

    # deltas relative to default
    merged['observed_itl_delta_ms'] = merged['mean_itl_ms'] - merged['itl_default_ms']
    merged['predicted_itl_delta_ms'] = merged['predicted_comm_ms'] - merged['predicted_comm_default_ms']

    # calculate percent error for the juist configs
    mask = (merged["interconnect"] != "default") & (merged["tp"] > 1)

    pct_error = (
        (merged['predicted_itl_delta_ms'] - merged['observed_itl_delta_ms']).abs()
        / merged['observed_itl_delta_ms'].abs() * 100)

    merged["pct_error"] = np.where(mask, pct_error, np.nan)

    return merged

# def plot_message_size_vs_concurrency(models=MODELS):
#     """ message size per AllReduce as concurrency grows, per model """
#     fig, ax = plt.subplots(figsize=(6, 3.5))

#     colors = ["#E85C4C", "#4C9BE8", "#5CB85C"]
#     for ((name, model), color) in zip(models.items(), colors):
#         sizes = [allreduce_message_size_bytes(b, model['hidden_size']) / 1024
#                  for b in CONCURRENCIES]
#         ax.plot(CONCURRENCIES, sizes, label=name, color=color, marker='o', markersize=2)

#     ax.set_xscale('log', base=2)
#     ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
#     ax.set_yscale('log', base=2)
#     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f} KB' if x < 1024 else f'{x/1024:.0f} MB'))
#     ax.set_xlabel('Concurrency (batch size)')
#     ax.set_ylabel('AllReduce message size')
#     ax.set_title('AllReduce Message Size vs Concurrency', pad=10)
#     ax.legend(fontsize=7)
#     ax.grid(True, axis='y', alpha=0.3)
#     plt.tight_layout()
#     return fig


def main(args):

    results_csv_path = Path(args.results_csv)
    nccl_csv_path = Path(args.nccl_csv)

    if not results_csv_path.exists():
        raise FileNotFoundError(f"results csv not found {results_csv_path}")
    if not nccl_csv_path.exists():
        raise FileNotFoundError(f"nccl csv not found {nccl_csv_path}")
    
    results_df = pd.read_csv(results_csv_path)
    results_df = results_df[(results_df['max_concurrency'] > 1) & 
                            (results_df['gpu_type'] == 'a100')]
    nccl_df = pd.read_csv(nccl_csv_path)

    combined_data = merge_measurements_onto_df(results_df, nccl_df)
    combined_data = add_predicted_itl_deltas(combined_data)

    output_path = results_csv_path.parent / "data_with_allreduce_stuff.csv"
    combined_data.to_csv(output_path, index=False)
    print(f"data with matching allreduce measurements saved to: {output_path}")

    # fig = plot_message_size_vs_concurrency()
    # plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "results_csv",
        type=str,
        help="path to results csv"
    )

    parser.add_argument(
        "nccl_csv",
        type=str,
        help="path to nccl test measurement csv"
    )

    args = parser.parse_args()

    main(args)


