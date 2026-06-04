#%%
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

def compute_derived_metrics(df):
    """ compute derived communication metrics by comparing multi-GPU configs to TP=1 baseline """

    # add normalized concurrency
    df['concurrency_per_gpu'] = df['max_concurrency'] / df['tp']

    # tp=1 baseline
    tp1_baseline = (
        df[df['tp'] == 1]
        .groupby(['gpu_type', 'model_name', 'concurrency_per_gpu'])
        .agg({
            'mean_itl_ms': 'mean',
            'output_throughput': 'mean',
            'mean_ttft_ms': 'mean',
            'mean_e2el_ms': 'mean',
            'p99_itl_ms': 'mean',
            'p90_itl_ms': 'mean',
        })
        .rename(columns={
            'mean_itl_ms': 'itl_tp1',
            'output_throughput': 'throughput_tp1',
            'mean_ttft_ms': 'ttft_tp1',
            'mean_e2el_ms': 'e2el_tp1',
            'p99_itl_ms': 'p99_itl_tp1',
            'p90_itl_ms': 'p90_itl_tp1',
        })
        .reset_index()
    )

    # default interconnect baseline
    default_baseline = (
        df[df['interconnect'] == 'default']
        .groupby(['gpu_type', 'model_name', 'tp', 'concurrency_per_gpu'])
        .agg({
            'mean_itl_ms': 'mean',
            'output_throughput': 'mean',
            'mean_ttft_ms': 'mean',
            'mean_e2el_ms': 'mean',
            'p99_itl_ms': 'mean',
            'p90_itl_ms': 'mean',
        })
        .rename(columns={
            'mean_itl_ms': 'itl_default',
            'output_throughput': 'throughput_default',
            'mean_ttft_ms': 'ttft_default',
            'mean_e2el_ms': 'e2el_default',
            'p99_itl_ms': 'p99_itl_default',
            'p90_itl_ms': 'p90_itl_default',
        })
        .reset_index()
    )
    # load multi-gpu data
    configs = df[df['tp'] > 1].copy()

    merged = configs.merge(
        tp1_baseline,
        on=['concurrency_per_gpu', 'model_name', 'gpu_type'],
        how='inner'
    )

    merged = merged.merge(
        default_baseline,
        on=['concurrency_per_gpu', 'model_name', 'tp', 'gpu_type'],
        how='left'
    )

    # tp=1 baseline derived metrics
    merged['itl_delta'] = merged['mean_itl_ms'] - merged['itl_tp1']
    merged['throughput_efficiency'] = (
        merged['output_throughput'] / 
        (merged['tp'] * merged['throughput_tp1'])
    )
    merged['itl_slowdown'] = merged['mean_itl_ms'] / merged['itl_tp1']
    merged['ttft_slowdown'] = merged['mean_ttft_ms'] / merged['ttft_tp1']
    merged['e2el_slowdown'] = merged['mean_e2el_ms'] / merged['e2el_tp1']

    # tail latency
    merged['p99_itl_slowdown'] = merged['p99_itl_ms'] / merged['p99_itl_tp1']
    merged['p90_itl_slowdown'] = merged['p90_itl_ms'] / merged['p90_itl_tp1']

    merged['itl_tail_ratio_p99'] = merged['p99_itl_ms'] / merged ['mean_itl_ms']
    merged['itl_tail_ratio_p90'] = merged['p90_itl_ms'] / merged ['mean_itl_ms']

    # interconnect degradation relative to default
    merged['itl_interconnect_degradation'] = merged['mean_itl_ms'] / merged['itl_default']
    merged['ttft_interconnect_degradation'] = merged['mean_ttft_ms'] / merged['ttft_default']
    merged['e2el_interconnect_degradation'] = merged['mean_e2el_ms'] / merged['e2el_default']
    merged['throughput_interconnect_efficiency'] = merged['output_throughput'] / merged['throughput_default']
    merged['p99_itl_interconnect_degradation'] = merged['p99_itl_ms'] / merged['p99_itl_default']
    merged['p90_itl_interconnect_degradation'] = merged['p90_itl_ms'] / merged['p90_itl_default']

    return merged

if __name__ == "__main__":

    results_dir = "./results/"

    # load sequence length sweep data
    seqlen_df = pd.read_csv(f'{results_dir}seqlen-sweep/averaged_seqlen_sweep_results.csv')
    # load concurrency sweep data
    concurrency_df = pd.read_csv(f'{results_dir}concurrency-sweep/averaged_concurrency_sweep_results.csv')

    # # prefill analysis
    # for gpu_type in ['a100', 'v100']:
    #     print(f"{gpu_type.upper()} ESTIMATES")
    #     for osl in [128,256,512,1024,2048]:
    #         compute_prefill_rate(seqlen_df, gpu_type=gpu_type, osl=osl)

    derived = compute_derived_metrics(concurrency_df)

    out_path = './results/concurrency-sweep/derived_metrics.csv'
    derived.to_csv(out_path, index=False)
    print(f"saved derived_metrics.csv ({len(derived)} rows)")

