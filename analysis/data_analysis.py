import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_prefill_rate(df, gpu_type, osl=512):
    data = df[(df['gpu_type'] == gpu_type) & 
              (df['random_output_len'] == osl)]
    
    x = data['random_input_len'].values.reshape(-1,1)
    y = data['mean_ttft_ms'].values

    reg = LinearRegression().fit(x, y)

    print(f'    Fixed OSL: {osl}')
    print(f"    Prefill rate: {reg.coef_[0]:.4f} ms/token\n")

def estimate_allocated_cache_size(vram, gpu_mem_util, num_model_params_B):
    # KV cache = VRAM × gpu_mem_util − model weights − (activations + overhead)
    dtype_size = 2
    allocated_size = vram * gpu_mem_util - dtype_size * num_model_params_B
    
    print("------------------------------------------------")
    print(f"with total VRAM {vram} |  gpu_mem_utilization = {gpu_mem_util} | size {num_model_params_B}B model")
    print(f"estimated allocated KV cache size = {allocated_size - (0.1*vram):.1f} - {allocated_size - (0.05*vram):.1f}")
    print("------------------------------------------------")

def estimate_max_sequences(cache_size, context_size, nlayers, nheads, head_dim):
    dtype_size = 2
    mem_per_token = 2 * nlayers * nheads * head_dim * dtype_size 
    
    max_batch = cache_size * 10**9 / (mem_per_token * context_size)
    print("------------------------------------------------")
    print(f"with KV cache size of {cache_size}GB")
    print(f"model with {nlayers} layers and {nheads} heads of dim {head_dim}")
    print(f"the kv cache's capacity is {max_batch:.1f} concurrent requests")
    print("------------------------------------------------")

def compute_derived_metrics(df):
    """ compute derived communication metrics by comparing multi-GPU configs to TP=1 baseline """

    all_derived = []

    baseline = (
        df[df['tp'] == 1]
        .groupby(['gpu_type', 'model_id', 'max_concurrency'])
        .agg({
            'mean_itl_ms': 'mean',
            'output_throughput': 'mean',
            'mean_ttft_ms': 'mean',
            'mean_e2el_ms': 'mean'
        })
        .rename(columns={
            'mean_itl_ms': 'itl_tp1',
            'output_throughput': 'throughput_tp1',
            'mean_ttft_ms': 'ttft_tp1',
            'mean_e2el_ms': 'e2el_tp1'
        })
        .reset_index()
    )

    # load multi-gpu data
    configs = df[df['tp'] > 1].copy()

    merged = configs.merge(
        baseline,
        on=['max_concurrency', 'model_id', 'gpu_type'],
        how='inner'
    )

    # derived metrics
    merged['itl_delta'] = merged['mean_itl_ms'] - merged['itl_tp1']
    merged['throughput_efficiency'] = (
        merged['output_throughput'] / 
        (merged['tp'] * merged['throughput_tp1'])
    )

    merged['itl_slowdown'] = merged['mean_itl_ms'] / merged['itl_tp1']
    merged['ttft_slowdown'] = merged['mean_ttft_ms'] / merged['ttft_tp1']
    merged['e2el_slowdown'] = merged['mean_e2el_ms'] / merged['e2el_tp1']

    return merged


if __name__ == "__main__":

    results_dir = "./resultscopy/"

    # load sequence length sweep data
    seqlen_df = pd.read_csv(f'{results_dir}seqlen-sweep/averaged_seqlen_sweep_results.csv')
    # load concurrency sweep data
    concurrency_df = pd.read_csv(f'{results_dir}concurrency-sweep/averaged_concurrency_sweep_results.csv')

    # prefill analysis
    for gpu_type in ['a100', 'v100']:
        print(f"{gpu_type.upper()} ESTIMATES")
        for osl in [128,256,512,1024,2048]:
            compute_prefill_rate(seqlen_df, gpu_type=gpu_type, osl=osl)

    estimate_allocated_cache_size(vram=64, gpu_mem_util=0.85, num_model_params_B=8 )
    estimate_allocated_cache_size(vram=40, gpu_mem_util=0.85, num_model_params_B=8 )
    estimate_allocated_cache_size(vram=80, gpu_mem_util=0.85, num_model_params_B=8 )
    estimate_allocated_cache_size(vram=160, gpu_mem_util=0.85, num_model_params_B=8 )


    estimate_max_sequences(46, (512+256), 32, 8, 128)
    estimate_max_sequences(15, (512+256), 32, 8, 128)
    estimate_max_sequences(35, (512+256), 32, 8, 128)
    estimate_max_sequences(100, (512+256), 32, 8, 128)

    derived = compute_derived_metrics(concurrency_df)

    out_path = './resultscopy/concurrency-sweep/derived_metrics.csv'
    derived.to_csv(out_path, index=False)
    print(f"saved derived_metrics.csv ({len(derived)} rows)")

