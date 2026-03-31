import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_prefill_rate(df, gpu_type, osl=512):
    data = df[(df['gpu_type'] == gpu_type) & 
              (df['random_output_len'] == osl)]
    
    x = data['random_input_len'].values.reshape(-1,1)
    y = data['mean_ttft_ms'].values

    reg = LinearRegression().fit(x, y)

    y_pred = reg.predict(x)
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
    

# load sequence length sweep data
seqlen_df = pd.read_csv('./results/single-gpu/seqlen-sweep/averaged_seqlen_sweep_results.csv')
# load baseline concurrency sweep data
concurrency_df = pd.read_csv('./results/single-gpu/concurrency-sweep/baseline/averaged_concurrency_sweep_baseline_results.csv')
# load fine-grained concurrency sweep data
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

