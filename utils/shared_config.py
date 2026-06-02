import numpy as np

# model configs
MODELS = {
    "llama_1b": {
        "num_layers": 16,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "num_model_params_B": 1,
    },
    "llama_8b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "num_model_params_B": 8,
    },
    "llama_13b": {
        "num_layers": 40,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_kv_heads": 40,
        "num_model_params_B": 13,
    }
}

CONCURRENCIES = np.array([4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512,
                          640, 768, 1024, 1280, 1536, 1792, 2048])

# interconnect configs
INTERCONNECTS = {
    "default": {
        "label_a100": "NVLink",
        "label_v100": "PCIe",
        "color": "#E85C4C",
    },
    "p2p_disabled": {
        "label": "SHM",
        "color": "#9B59B6",
    },
    "network_socket": {
        "label": "Socket",
        "color": "#E8A84C",
    },
    "pcie_v100_baseline": {
        "label": "PCIe (V100)",
        "color": "#4C9BE8",
    },
}

TP_COLORS = {
    1: "#4C9BE8", 
    2: "#E85C4C", 
    4: "#5CB85C"
}

# --------------------------------------------------------------------------
# series for data organization
# --------------------------------------------------------------------------
def series_entry(label, color, linestyle='-', marker='o', **filters):
    """ data series defined by its filters """
    return {
        "filters": filters,
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
    }

def build_tp_series(gpu_type, model_name):
    """ TP=1,2,4 series using default interconnect """
    series = []
    for tp in [1,2,4]:
        series.append(series_entry(
            label=f"TP={tp}",
            color=TP_COLORS[tp],
            gpu_type=gpu_type,
            model_name=model_name,
            interconnect='default',
            tp=tp
        ))
    return series

def build_interconnect_series(gpu_type, model_name, tp=2):
    """ interconnect series including TP=1 baseline  """
    if gpu_type == 'a100':
        default_label = INTERCONNECTS['default']['label_a100'] 
    else:
        default_label = INTERCONNECTS['default']['label_v100']
    series = []
    series.append(series_entry(
        label=f"Single-GPU (TP=1)",
        color=TP_COLORS[1],
        gpu_type=gpu_type,
        model_name=model_name,
        interconnect='default',
        tp=1
    ))
    series.append(series_entry(
        label=f"{default_label} (TP={tp})",
        color=INTERCONNECTS['default']['color'],
        gpu_type=gpu_type,
        model_name=model_name,
        interconnect='default',
        tp=tp
    ))
    series.append(series_entry(
        label=f"SHM (TP={tp})",
        color=INTERCONNECTS['p2p_disabled']['color'],
        gpu_type=gpu_type,
        model_name=model_name,
        interconnect='p2p_disabled',
        tp=tp
    ))
    series.append(series_entry(
        label=f"Socket (TP={tp})",
        color=INTERCONNECTS['network_socket']['color'],
        gpu_type=gpu_type,
        model_name=model_name,
        interconnect='network_socket',
        tp=tp
    ))
    return series

