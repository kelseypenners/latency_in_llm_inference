#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# load data
serve_df = pd.read_csv('./results/single-gpu/serve_results.csv')
latency_thruput_df = pd.read_csv('./results/single-gpu/latency_throughput_results.csv')

""" COMPARISON ALL METRICS """
def plot_all_metrics(df):
    # set up plotting style and figure
    plt.style.use('classic')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Llama-3.2-1B Single-GPU Baseline: A100 vs V100', fontsize=16, fontweight='bold')

    # split gpu data respectively
    a100_data = serve_df[serve_df['GPU'] == 'a100']
    v100_data = serve_df[serve_df['GPU'] == 'v100']

    # labels for plots
    a100_data = a100_data.copy()
    v100_data = v100_data.copy()
    a100_data['config'] = a100_data['ISL'].astype(str) + ' / ' + a100_data['OSL'].astype(str)
    v100_data['config'] = v100_data['ISL'].astype(str) + ' / ' + v100_data['OSL'].astype(str)

    # sort by ISL, then OSL
    a100_data = a100_data.sort_values(['ISL', 'OSL'])
    v100_data = v100_data.sort_values(['ISL', 'OSL'])

    # plot 1: ttft
    ax1 = axes[0, 0] 
    x = np.arange(len(a100_data))
    width = 0.35
    ax1.bar(x - width/2, a100_data['TTFT mean'], width, label='A100-40GB', alpha=0.8, color='firebrick')
    ax1.bar(x + width/2, v100_data['TTFT mean'], width, label='V100-32GB', alpha=0.8, color='cornflowerblue')
    ax1.set_xlabel('configuration (ISL/OSL)', fontweight='bold')
    ax1.set_ylabel('TTFT (ms)', fontweight='bold')
    ax1.set_title('Time to First Token (Prefill Phase)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(a100_data['config'], rotation=30, ha='right')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1500)
    ax1.grid(True, alpha=0.3)

    # plot 2: itl
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, a100_data['ITL mean'], width, label='A100-40GB', alpha=0.8, color='firebrick')
    ax2.bar(x + width/2, v100_data['ITL mean'], width, label='V100-32GB', alpha=0.8, color='cornflowerblue')
    ax2.set_xlabel('configuration (ISL/OSL)', fontweight='bold')
    ax2.set_ylabel('ITL (ms)', fontweight='bold')
    ax2.set_title('Inter-Token Latency (Decode Phase)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(a100_data['config'], rotation=30, ha='right')
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 70)
    ax2.grid(True, alpha=0.3)

    # plot 3: throughput
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, a100_data['output token throughput'], width, label='A100-40GB', alpha=0.8, color='firebrick')
    ax3.bar(x + width/2, v100_data['output token throughput'], width, label='V100-32GB', alpha=0.8, color='cornflowerblue')
    ax3.set_xlabel('configuration (ISL/OSL)', fontweight='bold')
    ax3.set_ylabel('tokens/Second', fontweight='bold')
    ax3.set_title('Output Token Throughput', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(a100_data['config'], rotation=30, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # plot 4: tpot
    ax4 = axes[1, 1]
    ax4.bar(x - width/2, a100_data['TPOT mean'], width, label='A100-40GB', alpha=0.8, color='firebrick')
    ax4.bar(x + width/2, v100_data['TPOT mean'], width, label='V100-32GB', alpha=0.8, color='cornflowerblue')
    ax4.set_xlabel('configuration (ISL/OSL)', fontweight='bold')
    ax4.set_ylabel('TPOT (ms)', fontweight='bold')
    ax4.set_title('Time Per Output Token', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_ylim(0, 70)
    ax4.set_xticklabels(a100_data['config'], rotation=30, ha='right')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./baseline_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved: baseline_comparison_all_metrics.png")


def plot_e2e_latency(df):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle('Llama 3.2-1B Single-GPU E2E Latency: A100 vs V100',
                fontsize=16, fontweight='bold')

    # split gpu data
    a100_data = latency_thruput_df[latency_thruput_df['GPU'] == 'a100'].copy()
    v100_data = latency_thruput_df[latency_thruput_df['GPU'] == 'v100'].copy()

    # create config labels
    a100_data['config'] = a100_data['ISL'].astype(str) + ' / ' + a100_data['OSL'].astype(str)
    v100_data['config'] = v100_data['ISL'].astype(str) + ' / ' + v100_data['OSL'].astype(str)

    # sort
    a100_data = a100_data.sort_values(['ISL', 'OSL'])
    v100_data = v100_data.sort_values(['ISL', 'OSL'])

    # plot e2e latency
    x = np.arange(len(a100_data))
    width = 0.35

    ax1.bar(x - width/2,
            a100_data['e2e latency mean'],
            width,
            label='A100-40GB',
            alpha=0.8,
            color='firebrick')

    ax1.bar(x + width/2,
            v100_data['e2e latency mean'],
            width,
            label='V100-32GB',
            alpha=0.8,
            color='cornflowerblue')

    ax1.set_xlabel('configuration (ISL / OSL)', fontweight='bold')
    ax1.set_ylabel('e2e latency (s)', fontweight='bold')
    ax1.set_title('E2E Latency across Configurations', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(a100_data['config'], rotation=30, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./baseline_e2e_sweeps.png', dpi=300, bbox_inches='tight')
    print("Saved: baseline_e2e_sweeps.png")

# %%
import matplotlib.pyplot as plt
import numpy as np

isl = np.array([2048, 512, 128])
osl = np.array([128, 256, 512, 1024])

# Example data
Z1 = np.array([
    [3209, 0, 8629, 0],
    [8543, 12753, 12872, 1296],
    [12580, 0, 0, 0]
])

Z2 = np.array([
    [1713, 0, 1651, 0],
    [4096, 4012, 3665, 3070],
    [6240, 0, 0, 0]
])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Base grid
xpos, ypos = np.meshgrid(np.arange(len(isl)),
                         np.arange(len(osl)),
                         indexing="ij")

xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

dx = dy = 0.35  # smaller width since we have two bars

# Offset for grouping
offset = 0.2

for i in range(len(isl)):
    for j in range(len(osl)):

        # First GPU
        ax.bar3d(i - offset, j, 0,
                 dx, dy, Z1[i, j],
                 alpha=0.5,
                 edgecolor='black',
                 linewidth=0.8,
                 shade=False,
                 color="red")

        # Second GPU
        ax.bar3d(i + offset, j, 0,
                 dx, dy, Z2[i, j],
                 alpha=0.5,
                 edgecolor='black',
                 linewidth=0.8,
                 shade=False,
                 color='blue')




# Axis labels
ax.set_xticks(np.arange(len(isl)))
ax.set_xticklabels(isl)

ax.set_yticks(np.arange(len(osl)))
ax.set_yticklabels(osl)

ax.set_xlabel("Input Sequence Length")
ax.set_ylabel("Output Sequence Length")
ax.set_zlabel("Throughput")

plt.show()

# %%

# Load data
a100_data = serve_df[serve_df['GPU'] == 'a100'].copy()
v100_data = serve_df[serve_df['GPU'] == 'v100'].copy()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: A100 TTFT heatmap
ax1 = axes[0, 0]
ttft_pivot_a100 = a100_data.pivot_table(values='TTFT mean', index='ISL', columns='OSL')
sns.heatmap(ttft_pivot_a100, annot=True, fmt='.0f', cmap='Greens', 
            cbar_kws={'label': 'TTFT (ms)'}, ax=ax1, linewidths=2, linecolor='white',
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_xlabel('OSl', fontweight='bold', fontsize=13)
ax1.set_ylabel('ISL', fontweight='bold', fontsize=13)
ax1.set_title('A100: Time to First Token', fontweight='bold', fontsize=14)
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=11)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=11)

# Top right: A100 ITL heatmap
ax2 = axes[0, 1]
itl_pivot_a100 = a100_data.pivot_table(values='ITL mean', index='ISL', columns='OSL')
sns.heatmap(itl_pivot_a100, annot=True, fmt='.1f', cmap='Blues',
            cbar_kws={'label': 'ITL (ms)'}, ax=ax2, linewidths=2, linecolor='white',
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_xlabel('OSL', fontweight='bold', fontsize=13)
ax2.set_ylabel('ISL', fontweight='bold', fontsize=13)
ax2.set_title('A100: Inter-Token Latency', fontweight='bold', fontsize=14)
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=11)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=11)

# Bottom left: V100 TTFT heatmap
ax3 = axes[1, 0]
ttft_pivot_v100 = v100_data.pivot_table(values='TTFT mean', index='ISL', columns='OSL')
sns.heatmap(ttft_pivot_v100, annot=True, fmt='.0f', cmap='Greens',
            cbar_kws={'label': 'TTFT (ms)'}, ax=ax3, linewidths=2, linecolor='white',
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})
ax3.set_xlabel('OSL', fontweight='bold', fontsize=13)
ax3.set_ylabel('ISL', fontweight='bold', fontsize=13)
ax3.set_title('V100: Time to First Token', fontweight='bold', fontsize=14)
ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=11)
ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=11)

# Bottom right: V100 ITL heatmap
ax4 = axes[1, 1]
itl_pivot_v100 = v100_data.pivot_table(values='ITL mean', index='ISL', columns='OSL')
sns.heatmap(itl_pivot_v100, annot=True, fmt='.1f', cmap='Blues',
            cbar_kws={'label': 'ITL (ms)'}, ax=ax4, linewidths=2, linecolor='white',
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})
ax4.set_xlabel('OSL', fontweight='bold', fontsize=13)
ax4.set_ylabel('ISL', fontweight='bold', fontsize=13)
ax4.set_title('V100: Inter-Token Latency', fontweight='bold', fontsize=14)
ax4.set_xticklabels(ax4.get_xticklabels(), fontsize=11)
ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=11)

plt.suptitle('GPU Comparison: ISL/OSL Impact on Performance Metrics', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# %%
