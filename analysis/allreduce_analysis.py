#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

from utils.shared_config import MODELS, INTERCONNECTS, CONCURRENCIES


def num_allreduce_per_step(num_layers):
    """ 2 AllReduce per transformer layer """
    return 2 * num_layers

def allreduce_message_size_bytes(batch_size, hidden_size, seq_len=1, dtype_bytes=2):
    """ size of tensor passed to AllReduce """
    # shape is [batch_size, seq_len, hidden_size]
    return batch_size * hidden_size * dtype_bytes * seq_len

def lookup_allreduce_latency(df, interconnect, message_bytes):
    filtered_df = df[df['interconnect'] == interconnect]
    if filtered_df.empty:
        return None
    row = filtered_df.iloc[0]
    sizes = json.loads(row['nccl_message_sizes'])
    latencies = json.loads(row['nccl_latency'])

    # linear interpolation based on measurements
    latency_us = np.interp(message_bytes, sizes, latencies)
    return latency_us

def total_comm_overhead_per_step(df, batch_size, model, interconnect):
    """ total AllReduce communication overhead in microseconds for one decode step """
    message_size = allreduce_message_size_bytes(batch_size, model["hidden_size"])
    num_ops = num_allreduce_per_step(model['num_layers'])
    latency = lookup_allreduce_latency(df, interconnect, message_size)
    if latency is None:
        return np.nan
    return num_ops * latency


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

def plot_total_comm_overhead_vs_concurrency(df, model_name, models=MODELS, interconnects=INTERCONNECTS):
    """ total communication overhead per decode step = n_allreduce * allreduce_latency """

    model = models[model_name]
    n_ops = num_allreduce_per_step(model['num_layers'])
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for name, cfg in interconnects.items():
        display_name = cfg.get('label', name)
        overhead = [total_comm_overhead_per_step(df, b, model, name) / 1000  # to ms
                    for b in CONCURRENCIES]
        ax.plot(CONCURRENCIES, overhead, label=display_name, color=cfg['color'], marker='o', markersize=2)

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Concurrency (batch size)')
    ax.set_ylabel('Predicted ITL overhead (ms)')
    ax.set_title(f'Total Communication Overhead per Decode Step\n{model_name}  |  {n_ops} AllReduces', pad=10)
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    pass
    #%%
    
    interconnect_measurements_path = "../results/interconnect_characterization/interconnect_characterization_frames.csv"
    derived = ""
    nccl_df = pd.read_csv(interconnect_measurements_path)

    fig1 = plot_message_size_vs_concurrency()
    plt.show()

    for model_name in MODELS:
        fig2 = plot_total_comm_overhead_vs_concurrency(nccl_df, model_name)
        plt.show()
        pass



# %%
