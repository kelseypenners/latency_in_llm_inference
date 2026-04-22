import sys
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_prefill_rate(df, gpu_type, osl=512):
    data = df[(df['gpu_type'] == gpu_type) & 
              (df['random_output_len'] == osl)]
    
    x = data['random_input_len'].values.reshape(-1,1)
    y = data['mean_ttft_ms'].values

    reg = LinearRegression().fit(x, y)

    print(f'GPU: {gpu_type}, fixed OSL: {osl}')
    print(f"Prefill rate: {reg.coef_[0]:.4f} ms/token\n")

def compute_avg_across_configs(df, gpu_type, metric):
    data = df[df['gpu_type'] == gpu_type][metric].dropna()

    print(f"GPU: {gpu_type} | Metric: {metric}")
    print(f"  mean:     {data.mean():.3f}")
    print(f"  std:      {data.std():.3f}")
    print(f"  min-max:  {data.min():.3f} -  {data.max():.3f}")
    print(f"  range:    {data.max() - data.min():.4f}")
    print(f"  configs:  {len(data)}")
    print()

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

def compute_derived_metrics(tp1_df, multi_df, label_col='label', allreduce_ops=64):
    """ compute derived communication metrics by comparing multi-GPU configs to TP=1 baseline """
    # observed_allreduce_latency_ms, relative_slowdown, degradation, observed_comm_overhead_ms

    # load tp1 baseline
    baseline = tp1_df[['max_concurrency', 'mean_itl_ms', 'output_throughput',
                       'mean_ttft_ms', 'mean_e2el_ms']].copy()
    baseline = baseline.rename(columns={
        'mean_itl_ms': 'itl_tp1',
        'output_throughput': 'throughput_tp1',
        'mean_ttft_ms': 'ttft_tp1',
        'mean_e2el_ms': 'e2el_tp1'
    })

    # load multi-gpu data
    configs = multi_df[['max_concurrency', label_col, 'mean_itl_ms', 'output_throughput',
                         'mean_ttft_ms', 'mean_e2el_ms']].copy()

    # merge on concurrency
    merged = configs.merge(baseline, on='max_concurrency', how='inner')

    # derived metrics
    delta_itl = merged['mean_itl_ms'] - merged['itl_tp1']
    merged['observed_comm_overhead_ms'] = delta_itl
    merged['observed_allreduce_latency_ms'] = delta_itl / allreduce_ops
    merged['relative_itl_slowdown'] = merged['mean_itl_ms'] / merged['itl_tp1']
    merged['relative_ttft_slowdown'] = merged['mean_ttft_ms'] / merged['ttft_tp1']
    merged['relative_e2el_slowdown'] = merged['mean_e2el_ms'] / merged['e2el_tp1']
    merged['degradation'] = delta_itl / merged['itl_tp1'] * 100

    # throughput efficiency
    tp_degree_map = {
        'mainsweep': 2,         # TP=2 NVLink
        'p2pdisabled': 2,       # TP=2 SHM
        'networkfallback': 2,   # TP=2 Socket
        'mainsweep_tp4':4 ,     # TP=4
    }
    merged['tp_degree'] = merged[label_col].map(tp_degree_map)
    merged['throughput_efficiency'] = (
        merged['output_throughput'] / (merged['tp_degree'] * merged['throughput_tp1'])
    )

    return merged


if __name__ == "__main__":
    # load sequence length sweep data
    seqlen_df = pd.read_csv('./results/single-gpu/seqlen-sweep/averaged_single_gpu_seqlen_sweep_results.csv')
    # load concurrency sweep data
    concurrency_df = pd.read_csv('./results/single-gpu/concurrency-sweep/fine-grained/averaged_concurrency_sweep_fine_grained_results.csv')

    # gpu-specific
    a100_data = seqlen_df[seqlen_df['gpu_type'] == 'a100'].copy()
    v100_data = seqlen_df[seqlen_df['gpu_type'] == 'v100'].copy()

    compute_prefill_rate(a100_data, gpu_type='a100', osl=128)
    compute_prefill_rate(a100_data, gpu_type='a100', osl=256)
    compute_prefill_rate(a100_data, gpu_type='a100', osl=512)
    compute_prefill_rate(a100_data, gpu_type='a100', osl=1024)
    compute_prefill_rate(a100_data, gpu_type='a100', osl=2048)

    compute_prefill_rate(v100_data, gpu_type='v100', osl=128)
    compute_prefill_rate(v100_data, gpu_type='v100', osl=256)
    compute_prefill_rate(v100_data, gpu_type='v100', osl=512)
    compute_prefill_rate(v100_data, gpu_type='v100', osl=1024)
    compute_prefill_rate(v100_data, gpu_type='v100', osl=2048)

    compute_avg_across_configs(a100_data, gpu_type='a100', metric='output_throughput')
    compute_avg_across_configs(a100_data, gpu_type='a100', metric='mean_itl_ms')

    compute_avg_across_configs(v100_data, gpu_type='v100', metric='output_throughput')
    compute_avg_across_configs(v100_data, gpu_type='v100', metric='mean_itl_ms')

    estimate_allocated_cache_size(vram=64, gpu_mem_util=0.85, num_model_params_B=8 )
    estimate_allocated_cache_size(vram=40, gpu_mem_util=0.85, num_model_params_B=8 )
    estimate_allocated_cache_size(vram=80, gpu_mem_util=0.85, num_model_params_B=8 )
    estimate_allocated_cache_size(vram=160, gpu_mem_util=0.85, num_model_params_B=8 )


    estimate_max_sequences(46, (512+256), 32, 8, 128)
    estimate_max_sequences(15, (512+256), 32, 8, 128)
    estimate_max_sequences(35, (512+256), 32, 8, 128)
    estimate_max_sequences(100, (512+256), 32, 8, 128)

    # multi-gpu derived metrics
    multigpu_path  = './results/multi-gpu/concurrency-sweep/a100/averaged_concurrency_sweep.csv'
    singlegpu_path = './results/single-gpu/concurrency-sweep/fine-grained/averaged_concurrency_sweep_fine_grained_results.csv'

    multigpu_raw  = pd.read_csv(multigpu_path)
    singlegpu_raw = pd.read_csv(singlegpu_path)

    tp1_df     = singlegpu_raw[singlegpu_raw['gpu_type'] == 'a100'].copy()

    # all multi-GPU configs in one dataframe
    multi_df   = multigpu_raw[multigpu_raw['label'].isin(
        ['mainsweep', 'networkfallback', 'p2pdisabled', 'mainsweep_tp4']
    )].copy()

    derived = compute_derived_metrics(tp1_df, multi_df)

    out_path = './results/multi-gpu/concurrency-sweep/a100/derived_metrics.csv'
    derived.to_csv(out_path, index=False)
    print(f"saved derived_metrics.csv ({len(derived)} rows)")

