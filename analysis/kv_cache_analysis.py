import pandas as pd
import argparse

from utils.shared_config import *

# --------------------------------------------------------------------------
# kv cache functions
# --------------------------------------------------------------------------
def kv_per_request_gb(
        num_layers,
        hidden_size, 
        num_attention_heads,
        num_kv_heads,
        kv_dtype_b=2,
        tp=1,   
        isl=512, osl=512,
        **kwargs):
    """ estimated kv cache memory (GB) of one request at mid-decode """

    head_dim = hidden_size // num_attention_heads
    kv_heads_per_gpu = num_kv_heads / tp
    avg_tokens = isl + osl / 2

    kv_bytes = (2 * num_layers * avg_tokens * kv_heads_per_gpu
                * head_dim * kv_dtype_b)

    return kv_bytes / 1e9

def available_kv_cache_gb(
        gpu_vram, 
        gpu_mem_util, 
        num_model_params_B,
        tp=1,
        dtype_b=2,
        activation_frac=0.1,
        **kwargs):
    """ estimated allocated available kv cache size (GB) """

    weight_gb_per_gpu = (num_model_params_B * dtype_b) / tp
    usable_mem = gpu_vram * gpu_mem_util
    activation_mem = activation_frac * gpu_vram

    return  usable_mem - weight_gb_per_gpu - activation_mem

def kv_saturation_point(
        gpu_vram, 
        gpu_mem_util, 
        num_model_params_B,
        num_layers,
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        tp=1,
        dtype_b=2,
        isl=512, osl=512,
        activation_frac=0.1,
        **kwargs):
    """ estimated analytical concurrency where kv cache is saturated """

    kv_per_req = kv_per_request_gb(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        kv_dtype_b=dtype_b,
        tp=tp,
        isl=isl, osl=osl
    )

    kv_available = available_kv_cache_gb(
        gpu_vram=gpu_vram,
        gpu_mem_util=gpu_mem_util,
        num_model_params_B=num_model_params_B,
        tp=tp,
        dtype_b=dtype_b,
        activation_frac=activation_frac,
    )
    
    return kv_available / kv_per_req

def saturation_points_across_tps(
        model_name, tps, gpu_vram, 
        gpu_mem_util=0.85, isl=512, osl=512,
        activation_frac=0.1):
    

    model_config = MODELS[model_name]
    rows = []

    for tp in tps:
        sat = kv_saturation_point(
            gpu_vram=gpu_vram,
            gpu_mem_util=gpu_mem_util,
            tp=tp,
            isl=isl, osl=osl,
            activation_frac=activation_frac,
            **model_config
        )
        rows.append({"tp": tp, "saturation_point": round(sat)})
    df = pd.DataFrame(rows)
    print(f"\n{model_name}  |   gpu_vram={gpu_vram}GB   |   ISL/OSL={isl}/{osl}")
    print(df.to_string(index=False))
    return df

def main(args):

    tps = [1,2,4]

    gpu_vram = args.gpu_vram
    activation_frac = args.activation_frac

    print(f"\nPREDICTED SATURATION POINTS FOR {gpu_vram}GB GPUS")
    print(f"    using ISL={args.isl} / OSL={args.osl}")

    if args.model:
        if args.model not in MODELS:
            print(f"error: model {args.model} not found in model config")
            return
        models_to_run = [args.model]
    else:
        models_to_run = list(MODELS.keys())


    for model_name in models_to_run:
        saturation_points_across_tps(model_name, tps=tps, gpu_vram=gpu_vram, 
                                     activation_frac=activation_frac, isl=args.isl, osl=args.osl)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        type=str,
        help='model name'
    )

    parser.add_argument(
        "--gpu-vram",
        type=int,
        help="number of GB VRAM of a single GPU"
    )

    parser.add_argument(
        "--isl",
        type=int,
        default=512,
        help="input sequence length in tokens"
    )

    parser.add_argument(
        "--osl",
        type=int,
        default=512,
        help="output sequence length in tokens"
    )

    parser.add_argument(
        "--activation-frac",
        type=float,
        default=0.1,
        help="fraction of VRAM used for activations"
    )

    args = parser.parse_args()
    main(args)