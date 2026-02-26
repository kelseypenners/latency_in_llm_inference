#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_utils import *

# load data
sweep_df = pd.read_csv('../results/single-gpu/23-02-2026/averaged_sweep_results.csv')
batch_sweep_df = pd.read_csv('../results/single-gpu/25-02-2026/averaged_sweep_results.csv')

# gpu-specific
a100_data = sweep_df[sweep_df['gpu_type'] == 'a100'].copy()
v100_data = sweep_df[sweep_df['gpu_type'] == 'v100'].copy()

def plot_input_output_heatmap(df, values='mean__ms', title='TTFT (ms)'):
    """ plot heatmap for input and output lengths and give value """
    pivot = df.pivot_table(values=values, index='random_output_len', columns='random_input_len')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Oranges', ax=ax, linecolor="white")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Input Sequence Length (ISL)')
    ax.set_ylabel('Output Sequence Length (OSL)')
    plt.tight_layout()

    return fig

def plot_ttft_vs_isl(df, osl=None):
    """ plot impact of ISL on TTFT """
    
    if osl == None:
        osl_values = sorted(df['random_output_len'].unique())
    else:
        osl_values = [osl]
    fig, ax = plt.subplots(figsize=(8,6))
    #osl_values = sorted(df['random_output_len'].unique())

    for osl in osl_values:
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'], marker='o', label=f"OSL={osl}")
    ax.set_title(f'Impact of Input Sequence Length (ITL) on TTFT')
    ax.set_xlabel('ISL (tokens)')
    ax.legend()
    ax.set_ylabel('TTFT (ms)')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    return fig

def plot_ttft_vs_osl_per_isl(df):
    """ plot impact of OSL on TTFT for each ISL config """

    isl_values = sorted(df['random_input_len'].unique())
    fig, axes = plt.subplots(1, len(isl_values), figsize=(4*len(isl_values), 4), sharey=True)

    for ax, isl in zip(axes, isl_values):
        data = df[df['random_input_len'] == isl]
        ax.plot(data['random_output_len'], data['mean_ttft_ms'], marker='o')
        ax.set_title(f'ISL={isl}')
        ax.set_xlabel('OSL (tokens)')
        ax.grid(True, alpha=0.4)
    
    axes[0].set_ylabel('TTFT (ms)')
    plt.suptitle('TTFT vs OSL (fixed ISL)', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_ttft_vs_isl_per_osl(df):
    """ plot impact of ISL on TTFT for each OSL config """

    osl_values = sorted(df['random_output_len'].unique())
    fig, axes = plt.subplots(1, len(osl_values), figsize=(4*len(osl_values), 4), sharey=True)

    for ax, osl in zip(axes, osl_values):
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_ttft_ms'], marker='o')
        ax.set_title(f'OSL={osl}')
        ax.set_xlabel('ISL (tokens)')
        ax.grid(True, alpha=0.4)
    
    axes[0].set_ylabel('TTFT (ms)')
    plt.suptitle('TTFT vs ISL (fixed OSL)', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_e2el_vs_osl(df, isl=None):
    """ plot impact of OSL on E2E latency """
    
    if isl == None:
        isl_values = sorted(df['random_input_len'].unique())
    else:
        isl_values = [isl]
    fig, ax = plt.subplots(figsize=(8,6))

    for isl in isl_values:
        data = df[df['random_input_len'] == isl]
        ax.plot(data['random_output_len'], data['mean_e2el_ms'], marker='o', label=f"ISL={isl}")
    ax.set_title(f'Impact of Output Sequence Length (OSL) on E2E Latency')
    ax.set_xlabel('OSL (tokens)')
    ax.legend()
    ax.set_ylabel('E2E Latency (ms)')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    return fig

def plot_e2el_vs_isl_per_osl(df):
    """ plot impact of ISL on TTFT for each OSL config """

    osl_values = sorted(df['random_output_len'].unique())
    fig, axes = plt.subplots(1, len(osl_values), figsize=(4*len(osl_values), 4), sharey=True)

    for ax, osl in zip(axes, osl_values):
        data = df[df['random_output_len'] == osl]
        ax.plot(data['random_input_len'], data['mean_e2el_ms'], marker='o')
        ax.set_title(f'OSL={osl}')
        ax.set_xlabel('ISL (tokens)')
        ax.grid(True, alpha=0.4)
    
    axes[0].set_ylabel('E2E Latency')
    plt.suptitle('E2E Latency vs ISL (fixed OSL)', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_metrics_vs_batch(df, metric, label):
    # TTFT, ITL, E2EL
    
    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(df['max_concurrency'], df[metric], marker='o')
    ax.set_title(f'Impact of Batch Size on {label}')
    ax.set_xlabel('Batch Size')
    ax.legend()
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    return fig

# %%

# TTFT vs ISL (across all OSLs)
fig = plot_ttft_vs_isl(a100_data)

# to look at trends for ISL and OSL 
fig = plot_ttft_vs_isl_per_osl(a100_data)
fig = plot_ttft_vs_osl_per_isl(a100_data)

# heatmaps for all metrics
fig = plot_input_output_heatmap(a100_data, values='output_throughput', title='Output Throughput (tokens/s)')
fig = plot_input_output_heatmap(a100_data, values='total_token_throughput', title='Total Token Throughput (tokens/s)')
fig = plot_input_output_heatmap(a100_data, values='mean_e2el_ms', title='Average E2E Latency (ms)')
fig = plot_input_output_heatmap(a100_data, values='mean_itl_ms', title='ITL (ms)')
fig = plot_input_output_heatmap(a100_data, values='mean_tpot_ms', title='TPOT (ms)')
fig = plot_input_output_heatmap(a100_data, values='mean_ttft_ms', title='TTFT (ms)')

# E2EL vs ISL and OSL
fig = plot_e2el_vs_osl(a100_data)
fig = plot_e2el_vs_isl_per_osl(a100_data)

# batch size versus metrics
fig = plot_metrics_vs_batch(batch_sweep_df, "mean_ttft_ms", "TTFT (ms)")
fig = plot_metrics_vs_batch(batch_sweep_df, "mean_itl_ms", "ITL (ms)")
fig = plot_metrics_vs_batch(batch_sweep_df, "mean_tpot_ms", "TPOT (ms)")
fig = plot_metrics_vs_batch(batch_sweep_df, "mean_e2el_ms", "E2EL (ms)")
fig = plot_metrics_vs_batch(batch_sweep_df, "output_throughput", "Output Throughput")  
fig = plot_metrics_vs_batch(batch_sweep_df, "total_token_throughput", "Total Token Throughput")  

# %%
