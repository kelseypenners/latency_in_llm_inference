#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_utils import *

# load data
sweep_df = pd.read_csv('../results/single-gpu/23-02-2026/averaged_sweep_results.csv')

# gpu-specific
a100_data = sweep_df[sweep_df['gpu_type'] == 'a100'].copy()
v100_data = sweep_df[sweep_df['gpu_type'] == 'v100'].copy()

def plot_input_output_heatmap(df, values='mean__ms', title='TTFT (ms)'):
    """ plot heatmap for input and output lengths and give value """
    pivot = df.pivot_table(values=values, index='random_output_len', columns='random_input_len')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Oranges', ax=ax)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Input Sequence Length (ISL)')
    ax.set_ylabel('Output Sequence Length (OSL)')
    plt.tight_layout()

    return fig

def plot_ttft_vs_isl(df, osl=128):
    """ plot impact of ISL on TTFT"""

    fixed_osl = df[df['random_output_len'] == osl].copy()
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fixed_osl['random_input_len'], fixed_osl['mean_ttft_ms'], marker='o')
    ax.set_title(f'Impact of ISL on TTFT (fixed OSL={osl})')
    ax.set_xlabel('Input Sequence Length (tokens)')
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

# %%
fig = plot_ttft_vs_isl(a100_data, osl=512)

# to look at trends for ISL and OSL 
fig = plot_ttft_vs_isl_per_osl(a100_data)
fig = plot_ttft_vs_osl_per_isl(a100_data)

# heatmaps
fig = plot_input_output_heatmap(a100_data, values='output_throughput', title='Output Throughput (tokens/s)')
fig = plot_input_output_heatmap(a100_data, values='total_token_throughput', title='Total Token Throughput (tokens/s)')
fig = plot_input_output_heatmap(a100_data, values='mean_e2el_ms', title='Average E2E Latency (ms)')


# %%
