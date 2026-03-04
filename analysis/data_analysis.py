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
    print(f"Prefill rate: {reg.coef_[0]:.4f} ms/token")
    print(f"Intercept: {reg.intercept_:.2f}")



# load data
sweep_df = pd.read_csv('../results/single-gpu/23-02-2026/averaged_sweep_results.csv')
batch_sweep_df = pd.read_csv('../results/single-gpu/25-02-2026/averaged_concurrency_sweep_results.csv')

batch_sweep_df = pd.read_csv('../results/single-gpu/27-02-2026/averaged_batch_sweep_results.csv')

# gpu-specific
a100_data = sweep_df[sweep_df['gpu_type'] == 'a100'].copy()

compute_prefill_rate(a100_data, gpu_type='a100', osl=128)
compute_prefill_rate(a100_data, gpu_type='a100', osl=256)
compute_prefill_rate(a100_data, gpu_type='a100', osl=512)
compute_prefill_rate(a100_data, gpu_type='a100', osl=1024)
compute_prefill_rate(a100_data, gpu_type='a100', osl=2048)