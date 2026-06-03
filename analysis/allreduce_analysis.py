#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    """ lookup AllReduce lateny measurement based on config """
    filtered_df = filter_df(df, interconnect=interconnect, tp=tp, server=server)
    if filtered_df.empty:
        return None
    row = filtered_df.iloc[0]
    sizes = json.loads(row['nccl_message_sizes'])
    latencies = json.loads(row['nccl_latency'])

    # linear interpolation based on measurements
    latency_us = np.interp(message_bytes, sizes, latencies)
    return latency_us

def predicted_comm_overhead_per_step(df, server, batch_size, model, interconnect, tp=2):
    """ total AllReduce communication overhead in milliseconds for one decode step """
    message_size = allreduce_message_size_bytes(batch_size, model["hidden_size"])
    num_ops = num_allreduce_per_step(model['num_layers'])
    latency = lookup_allreduce_latency(df, server=server, interconnect=interconnect, 
                                       tp=tp, message_bytes=message_size)
    if latency is None:
        return np.nan
    return (num_ops * latency) / 1000 # convert to milliseconds

def merge_measurements_onto_derived(derived_df, nccl_df):
    """ add interconnect measurement data to derived dataframe """
    df_copy = derived_df.copy()
    predicted_comm_ms = []

    model_name_to_cfg = {cfg["model_name"]: cfg for cfg in MODELS.values() if "model_name" in cfg}

    for _, row in df_copy.iterrows():
        row_model_name = row['model_name']
        
        if row_model_name not in model_name_to_cfg:
            predicted_comm_ms.append(np.nan)
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
        
    df_copy['predicted_comm_ms'] = predicted_comm_ms
    return df_copy


def plot_message_size_vs_concurrency(models=MODELS):
    """ message size per AllReduce as concurrency grows, per model """
    fig, ax = plt.subplots(figsize=(6, 3.5))

    colors = ["#E85C4C", "#4C9BE8", "#5CB85C"]
    for ((name, model), color) in zip(models.items(), colors):
        sizes = [allreduce_message_size_bytes(b, model['hidden_size']) / 1024
                 for b in CONCURRENCIES]
        ax.plot(CONCURRENCIES, sizes, label=name, color=color, marker='o', markersize=2)

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f} KB' if x < 1024 else f'{x/1024:.0f} MB'))
    ax.set_xlabel('Concurrency (batch size)')
    ax.set_ylabel('AllReduce message size')
    ax.set_title('AllReduce Message Size vs Concurrency', pad=10)
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_total_comm_overhead_vs_concurrency(df, server, model_name, tp=2, models=MODELS, interconnects=INTERCONNECTS):
    """ total communication overhead per decode step = n_allreduce * allreduce_latency """

    model = models[model_name]
    n_ops = num_allreduce_per_step(model['num_layers'])
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for name, cfg in interconnects.items():
        display_name = cfg.get('label', name)
        overhead = [predicted_comm_overhead_per_step(df, server=server, batch_size=b, model=model, interconnect=name, tp=2)
                    for b in CONCURRENCIES]
        ax.plot(CONCURRENCIES, overhead, label=display_name, color=cfg['color'], marker='o', markersize=2)

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Concurrency (batch size)')
    ax.set_ylabel('Predicted ITL overhead (ms)')
    ax.set_title(f'Predicted Communication Overhead per Decode Step\n{model_name}  | TP={tp} | {n_ops} AllReduces per step', pad=10)
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_expected_vs_observed_overhead(series, df, title, subtitle_info=None):

    fig, ax = plt.subplots(figsize=(4.5, 3), layout="constrained")

    for s in series:
        plot_df = filter_df(df, **s['filters']).sort_values('max_concurrency')
        if plot_df.empty:
            continue
            
        # plot observed change in ITL 
        ax.plot(plot_df['max_concurrency'], plot_df['itl_delta'],
                color=s['color'], marker=s['marker'], linestyle=s['linestyle'],
                label=f"{s['label']} (Observed Delta ITL)")
        
        # plot mathematical prediction based on AllReduce measurements
        ax.plot(plot_df['max_concurrency'], plot_df['predicted_comm_ms'],
                color=s['color'], linestyle='--', alpha=0.7, marker='x', markersize=0,
                label=f"{s['label']} (Isolated NCCL)")

    ax.set_title(title, pad=15)
    ax.set_xlabel('Concurrency (Batch Size)')
    ax.set_ylabel('Overhead Cost (ms)')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=6, loc='upper left')
    
    if subtitle_info:
        subtitle(ax, **subtitle_info)
        
    return fig

if __name__ == "__main__":
    pass
    #%%
    results_dir = "../results/"

    # load data
    derived_metrics_path = f"{results_dir}concurrency-sweep/derived_metrics.csv"
    derived = pd.read_csv(derived_metrics_path)
    derived = derived[(derived['max_concurrency'] > 1) & (derived['gpu_type'] == 'a100')] 

    interconnect_measurements_path = f"{results_dir}interconnect_characterization/interconnect_characterization_frames.csv"
    nccl_df = pd.read_csv(interconnect_measurements_path)

    combined_data = merge_measurements_onto_derived(derived_df=derived, nccl_df=nccl_df)

    # generate a100 interconnect series without single GPU baseline
    a100_interconnect_series = build_interconnect_series(gpu_type='a100', model_name="llama-3.1-8b", tp=2)
    a100_interconnect_series = [s for s in a100_interconnect_series if s['filters'].get('tp', 1) > 1]

    sub_a100_8b = dict(gpu='shy-fec (A100s)', 
           model='Llama-3.1-8B', 
           config='ISL/OSL = 512/512')

    fig1 = plot_message_size_vs_concurrency()
    plt.show()

    fig2 = plot_expected_vs_observed_overhead(
        series=a100_interconnect_series, 
        df=combined_data,
        title="Expected vs. Observed Communication Overhead",
        subtitle_info=sub_a100_8b
    )
    plt.show()

  

    for model_name in MODELS:
        fig2 = plot_total_comm_overhead_vs_concurrency(nccl_df, server="shy-fec", model_name=model_name)
        plt.show()



# %%
