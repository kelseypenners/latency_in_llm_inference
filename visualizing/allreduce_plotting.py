#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import json
import matplot2tikz

from analysis.allreduce_analysis import *
from utils.plotting import *
from utils.shared_config import *

set_figure_style()

SATURATION_POINTS = {
    ("llama-3.2-1b", 1): 1113,
    ("llama-3.2-1b", 2): 2305,
    ("llama-3.2-1b", 4): 4689,
    ("llama-3.1-8b", 1): 139,
    ("llama-3.1-8b", 2): 437,
    ("llama-3.1-8b", 4): 1033,
    ("llama-2-13b", 1): 6,
    ("llama-2-13b", 2): 54,
    ("llama-2-13b", 4): 149,
}

results_dir = "../results/"
combined_data = pd.read_csv(f"{results_dir}concurrency-sweep/data_with_allreduce_stuff.csv")
nccl_df = pd.read_csv(f"{results_dir}interconnect_characterization/interconnect_characterization_frames.csv")

combined_data.to_csv("data_with_allreduce_stuff_saturation_cap.csv", index=False)

def plot_lat_vs_concurrency(series, df, title, subtitle_info=None):

    fig, ax = plt.subplots(figsize=(4.5, 3), layout="constrained")

    for s in series:
        plot_df = filter_df(df, **s['filters']).sort_values('max_concurrency')
        if plot_df.empty:
            continue
        
        # plot mathematical prediction based on AllReduce measurements
        ax.plot(plot_df['max_concurrency'], plot_df['allreduce_lat_ms'],
                color=s['color'], linestyle=s['linestyle'], markersize=0,
                label=f"{s['label']}")

    ax.set_title(title, pad=15)
    ax.set_xlabel('Concurrent Requests')
    ax.set_ylabel('AllReduce Latency per Op (ms)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    yticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=6, loc='upper left')
    
    if subtitle_info:
        subtitle(ax, **subtitle_info)

def plot_allreduce_lat_vs_message_size(df, server="shy-fec", tp_degrees=[2, 4], interconnects=INTERCONNECTS):
    """ plot NCCL AllReduce latency against message size for different interconnects """
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # min and max decode message sizes we've seen so far
    min_size = 4096
    max_size =  16777216

    # map tp degrees to linestyles
    tp_styles = {
        2: {'linestyle': '-', 'marker': 'o', 'label_suffix': '(TP=2)'},
        4: {'linestyle': '--', 'marker': 'D', 'label_suffix': '(TP=4)'}
    }

    for tp in tp_degrees:
        style = tp_styles.get(tp, {'linestyle': '-', 'marker': '.', 'label_suffix': f'(TP={tp})'})
        
        for name, cfg in interconnects.items():

            filtered_df = filter_df(df, interconnect=name, tp=tp, server=server)
            if filtered_df.empty:
                continue
                
            row = filtered_df.iloc[0]
            sizes = np.array(json.loads(row['nccl_message_sizes']))
            latencies = np.array(json.loads(row['nccl_latency']))

            mask = (sizes >= min_size) & (sizes <= max_size)
            if not np.any(mask):
                continue
                
            sizes_filtered = sizes[mask]
            latencies_filtered = latencies[mask] / 1000 # to milliseconds

            display_name = cfg.get('label', None)
            if display_name == None:
                display_name  = cfg.get('label_a100', name)
            display_name = f"{display_name} {style['label_suffix']}"
            
            ax.plot(
                sizes_filtered, 
                latencies_filtered, 
                label=display_name, 
                color=cfg['color'], 
                linestyle=style['linestyle'],
                marker=style['marker'], 
                markersize=3.5
            )

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)

    # custom formatter for bytes -> KB / MB
    def bytes_formatter(x, pos):
        if x >= 1024 * 1024:
            val = x / (1024 * 1024)
            return f'{val:g} MB'
        elif x >= 1024:
            val = x / 1024
            return f'{val:g} KB'
        else:
            return f'{x:g} B'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(bytes_formatter))
    
    yticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
    ax.set_xlabel('AllReduce Message Size (Bytes)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'NCCL AllReduce Latency vs Message Size', pad=10)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    return fig

def apply_saturation_cap(df, nccl_df, saturation_points):
    """ recompute allreduce_lat_ms and predicted_comm_ms with batch size
        at KV saturation point for each config """
    
    df = df.copy()
    model_name_to_cfg = {cfg["model_name"]: cfg for cfg in MODELS.values() 
                         if "model_name" in cfg}
    
    new_allreduce_lat = []
    new_predicted_comm = []
    
    for _, row in df.iterrows():
        row_model_name = row['model_name']
        if row_model_name not in model_name_to_cfg:
            new_allreduce_lat.append(np.nan)
            new_predicted_comm.append(np.nan)
            continue
        
        model_cfg = model_name_to_cfg[row_model_name]
        tp = row['tp']
        interconnect = row['interconnect']
        server = row.get('server', 'shy-fec')
        
        sat = saturation_points.get((row_model_name, tp), row['max_concurrency'])
        effective_batch = min(row['max_concurrency'], sat)
        
        message_size = allreduce_message_size_bytes(effective_batch, model_cfg["hidden_size"])
        lat_ms = lookup_allreduce_latency(
            nccl_df, server=server, interconnect=interconnect,
            tp=tp, message_bytes=message_size
        )
        new_allreduce_lat.append(lat_ms)
        
        num_ops = num_allreduce_per_step(model_cfg['num_layers'])
        new_predicted_comm.append((num_ops * lat_ms) if lat_ms is not None else np.nan)
    
    df['allreduce_lat_ms'] = new_allreduce_lat
    df['predicted_comm_ms'] = new_predicted_comm

    # recompute deltas
    cols_to_drop = [
        'itl_default_ms', 'predicted_comm_default_ms', 
        'observed_itl_delta_ms', 'predicted_itl_delta_ms', 'pct_error'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = add_predicted_itl_deltas(df)

    return df

def plot_allreduce_lat_vs_concurrency(df, model_name, gpu_type='a100', tp_degrees=[2, 4], 
                                       models=MODELS, interconnects=INTERCONNECTS):
    """ plot NCCL AllReduce latency per operation vs concurrency, one line per interconnect/tp """
    fig, ax = plt.subplots(figsize=(4, 3))

    tp_styles = {
        2: {'linestyle': '-',  'marker': 'o', 'label_suffix': '(TP=2)'},
        4: {'linestyle': '--', 'marker': 'o', 'label_suffix': '(TP=4)'},
    }

    model = MODELS[model_name]
    model_name = model['model_name']

    for tp in tp_degrees:
        style = tp_styles.get(tp, {'linestyle': '-', 'marker': '.', 'label_suffix': f'(TP={tp})'})

        for name, cfg in interconnects.items():
            filtered_df = filter_df(df, interconnect=name, tp=tp, gpu_type=gpu_type, model_name=model_name)
            filtered_df = filtered_df.sort_values('max_concurrency')
            if filtered_df.empty:
                continue

            display_name = cfg.get('label') or cfg.get('label_a100', name)
            label = f"{display_name} {style['label_suffix']}"

            ax.plot(
                filtered_df['max_concurrency'], filtered_df['allreduce_lat_ms'],
                label=label,
                color=cfg['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=3.5,
                picker=True
            )
            

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_yscale('log', base=10)
    yticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
    ax.set_xlabel('Concurrent Requests')
    ax.set_ylabel('AllReduce latency per op (ms)')
    ax.set_title(f'Measured NCCL AllReduce Latency\n{model['model_name']}', pad=10)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)

    def onpick(event):
        line = event.artist
        idx = event.ind[0]

        x = line.get_xdata()[idx]
        y = line.get_ydata()[idx]

        print(f"{line.get_label()}: x={x}, y={y:.4f} ms")

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.tight_layout()

    return fig

def plot_total_comm_overhead_vs_concurrency(df, model_name, tp=2, gpu_type='a100', models=MODELS, interconnects=INTERCONNECTS):
    """ total communication overhead per decode step = n_allreduce * allreduce_latency """

    fig, ax = plt.subplots(figsize=(5, 3.5))

    model = models[model_name]
    model_name = model['model_name']
    n_ops = model['num_layers'] * 2

    for name, cfg in interconnects.items():
        display_name = cfg.get('label', name)
        filtered_df = filter_df(df, interconnect=name, tp=tp, gpu_type=gpu_type, model_name=model_name)
        filtered_df = filtered_df.sort_values('max_concurrency')
        if filtered_df.empty:
            continue

        ax.plot(filtered_df['max_concurrency'], filtered_df['predicted_comm_ms'], label=display_name, color=cfg['color'], marker='o', markersize=2)

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
        ax.plot(plot_df['max_concurrency'], plot_df['mean_itl_ms'],
                color=s['color'], marker=s['marker'], linestyle=s['linestyle'],
                label=f"{s['label']} (Observed ITL)")
        
        # plot mathematical prediction based on AllReduce measurements
        ax.plot(plot_df['max_concurrency'], plot_df['predicted_comm_ms'],
                color=s['color'], linestyle='--', alpha=0.7, marker='x', markersize=0,
                label=f"{s['label']} (Isolated NCCL)")

    ax.set_title(title, pad=15)
    ax.set_xlabel('Concurrency (Batch Size)')
    ax.set_ylabel('ITL (ms)')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=6, loc='upper left')
    
    if subtitle_info:
        subtitle(ax, **subtitle_info)
        
    return fig

def plot_percent_allreduce_overhead(
    df, model_name="llama-3.1-8b", num_layers=32
):
    fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")

    # Filter for the target model
    model_df = df[df["model_name"] == model_name].copy()

    # Calculate pure communication overhead per decode step (2 * L * AR_lat)
    model_df["total_comm_ms"] = num_layers * 2 * model_df["allreduce_lat_ms"]

    # Compute percentage of total ITL consumed by AllReduce
    model_df["pct_comm_overhead"] = (
        model_df["total_comm_ms"] / model_df["mean_itl_ms"]
    ) * 100

    # Color palette for transports
    transport_colors = {
        "default": "#1f77b4",  # NVLink / PCIe P2P
        "p2p_disabled": "#ff7f0e",  # Host SHM
        "network_socket": "#d62728",  # TCP Socket
    }

    # Plot lines per (TP degree, Interconnect) pair
    for (tp, interconnect), group in model_df.groupby(["tp", "interconnect"]):
        if tp == 1:
            continue  # Skip single GPU (no AllReduce)

        print(f"{interconnect.upper()} (TP={tp}): min={group['pct_comm_overhead'].min():.2f}%, max={group['pct_comm_overhead'].max():.2f}%")
        group = group.sort_values("max_concurrency")
        color = transport_colors.get(interconnect, "gray")
        style = "-" if tp == 2 else "--"
        marker = "o" if tp == 2 else "s"

        label = f"{interconnect.upper()} (TP={tp})"

        ax.plot(
            group["max_concurrency"],
            group["pct_comm_overhead"],
            label=label,
            color=color,
            linestyle=style,
            marker=marker,
            markersize=4,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrency (Batch Size)")
    ax.set_ylabel("AllReduce Overhead (% of Total ITL)")
    ax.set_title(
        f"Communication Overhead Breakdown\n{model_name} (L={num_layers})",
        pad=10,
    )
    ax.set_ylim(0, 105)
    ax.grid(True, axis="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    return fig

#%%
combined_data = apply_saturation_cap(combined_data, nccl_df, SATURATION_POINTS)

# generate a100 interconnect series without single GPU baseline
a100_interconnect_series = build_interconnect_series(gpu_type='a100', model_name="llama-3.1-8b", tp=2)
a100_interconnect_series = [s for s in a100_interconnect_series if s['filters'].get('tp', 1) > 1]

a100_interconnect_series_tp4 = build_interconnect_series(gpu_type='a100', model_name="llama-3.1-8b", tp=4)
a100_interconnect_series_tp4 = [s for s in a100_interconnect_series_tp4 if s['filters'].get('tp', 1) > 1]

a100_tp_8b = build_tp_series('a100', 'llama-3.1-8b')
a100_tp_8b = [s for s in a100_tp_8b if s['filters'].get('tp', 1) > 1]
a100_tp_13b = build_tp_series('a100', 'llama-2-13b')
a100_tp_13b = [s for s in a100_tp_13b if s['filters'].get('tp', 1) > 1]

a100_tp_8b_p2pdisabled = build_tp_series('a100', 'llama-3.1-8b', interconnect = 'p2p_disabled')
a100_tp_8b_p2pdisabled = [s for s in a100_tp_8b_p2pdisabled if s['filters'].get('tp', 1) > 1]
a100_tp_13b_p2pdisabled = build_tp_series('a100', 'llama-2-13b', interconnect = 'p2p_disabled')
a100_tp_13b_p2pdisabled = [s for s in a100_tp_13b_p2pdisabled if s['filters'].get('tp', 1) > 1]

a100_tp_8b_socket = build_tp_series('a100', 'llama-3.1-8b', interconnect = 'network_socket')
a100_tp_8b_socket = [s for s in a100_tp_8b_socket if s['filters'].get('tp', 1) > 1]
a100_tp_13b_socket = build_tp_series('a100', 'llama-2-13b', interconnect = 'network_socket')
a100_tp_13b_socket = [s for s in a100_tp_13b_socket if s['filters'].get('tp', 1) > 1]



sub_a100_8b = dict(gpu='shy-fec (A100s)', 
        model='Llama-3.1-8B', 
        config='ISL/OSL = 512/512')
sub_a100_13b = dict(gpu='shy-fec (A100s)', 
        model='Llama-2-13B', 
        config='ISL/OSL = 512/512')

# a100_tp_8b = a100_tp_8b + a100_tp_8b_p2pdisabled
# a100_tp_13b = a100_tp_13b + a100_tp_13b_p2pdisabled

# a100_tp_8b = a100_tp_8b_socket
# a100_tp_13b = a100_tp_13b_socket


fig = plot_lat_vs_concurrency(
    series=a100_tp_8b,
    df=combined_data,
    title="Measured NCCL AllReduce Latency",
    subtitle_info=sub_a100_8b
)

fig = plot_lat_vs_concurrency(
    series=a100_tp_13b,
    df=combined_data,
    title="Measured NCCL AllReduce Latency",
    subtitle_info=sub_a100_13b
)

fig2 = plot_expected_vs_observed_overhead(
    series=a100_tp_8b, 
    df=combined_data,
    title="Expected vs. Observed Communication Overhead",
    subtitle_info=sub_a100_8b
)

fig2 = plot_expected_vs_observed_overhead(
    series=a100_tp_13b, 
    df=combined_data,
    title="Expected vs. Observed Communication Overhead",
    subtitle_info=sub_a100_13b
)

for model_name in MODELS:
    fig2 = plot_total_comm_overhead_vs_concurrency(combined_data, model_name=model_name)
    fig = plot_allreduce_lat_vs_concurrency(combined_data, model_name=model_name )
    fig_lat = plot_allreduce_lat_vs_message_size(nccl_df, server="shy-fec")
    break


# %%
fig = plot_percent_allreduce_overhead(combined_data, "llama-3.2-1b", num_layers=16)
fig = plot_percent_allreduce_overhead(combined_data, "llama-3.1-8b", num_layers=32)
fig = plot_percent_allreduce_overhead(combined_data, "llama-2-13b", num_layers=40)
# fig = plot_percent_allreduce_overhead(combined_data, "llama-3.2-1b", num_layers=32)
# %%
model="llama-3.1-8b"

a100_interconnect_series = build_interconnect_series(gpu_type='a100', model_name=model, tp=2)
a100_interconnect_series = [s for s in a100_interconnect_series if s['filters'].get('tp', 1) > 1]

a100_interconnect_series_tp4 = build_interconnect_series(gpu_type='a100', model_name=model, tp=4)
a100_interconnect_series_tp4 = [s for s in a100_interconnect_series_tp4 if s['filters'].get('tp', 1) > 1]

fig = plot_metric_vs_metric(a100_interconnect_series+a100_interconnect_series_tp4, combined_data, 
                            x_metric="predicted_itl_delta_ms", y_metric="observed_itl_delta_ms",
                            xlabel="predicted ITL Delta (ms)", ylabel="observed ITL Delta (ms)",
                            title="Predicted vs OBserved ITL Deltas"
                            )

fig = plot_metric_vs_concurrency(a100_interconnect_series+a100_interconnect_series_tp4, combined_data,
                                 metric="pct_error", ylabel="Percent Error",
                                 title="percent error")
# %%

def plot_delta_itl_vs_concurrency(df, model_name="llama-3.1-8b", tp=2):
    """
    Plots Observed Delta ITL vs Predicted Delta ITL directly on the Y-axis across Concurrency.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.8), layout="constrained")
    
    subset = df[(df["model_name"] == model_name) & (df["tp"] == tp)].copy()
    
    transport_colors = {
        "p2p_disabled": "#ff7f0e",  # host SHM
        "network_socket": "#d62728"  # TCP Socket
    }
    
    # filter out default baseline
    non_default_ic = [ic for ic in subset["interconnect"].unique() if ic != "default"]
    
    for ic in non_default_ic:
        ic_df = subset[subset["interconnect"] == ic].sort_values("max_concurrency")
        color = transport_colors.get(ic, "black")
        
        # observed delta ITL
        ax.plot(
            ic_df["max_concurrency"], 
            ic_df["observed_itl_delta_ms"],
            color=color, linestyle="-", marker="o", label=f"{ic.upper()} Observed ΔITL"
        )
        # predicted delta ITL
        ax.plot(
            ic_df["max_concurrency"], 
            ic_df["predicted_itl_delta_ms"],
            color=color, linestyle="--", marker="s", alpha=0.75, label=f"{ic.upper()} Predicted ΔITL"
        )
        
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Concurrency (Batch Size)", fontsize=9)
    ax.set_ylabel("ITL Delta Overhead Δ (ms)", fontsize=9)
    ax.set_title(f"Observed vs. Microbenchmark Predicted ΔITL\n{model_name.upper()} (TP={tp})", fontsize=10)
    ax.grid(True, axis="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    
    return fig, ax

# %%

fig_delta, _ = plot_delta_itl_vs_concurrency(combined_data, model_name="llama-3.1-8b", tp=2)
fig_delta, _ = plot_delta_itl_vs_concurrency(combined_data, model_name="llama-3.1-8b", tp=4)
# fig_delta, _ = plot_delta_itl_vs_concurrency(combined_data, model_name="llama-2-13b", tp=2)
# fig_delta, _ = plot_delta_itl_vs_concurrency(combined_data, model_name="llama-2-13b", tp=4)

plt.show()
# %%
