
from sklearn.linear_model import LinearRegression

def compute_prefill_rate(df, gpu_type, osl=512):
    data = df[(df['gpu_type'] == gpu_type) & 
              (df['random_output_len'] == osl)]
    
    x = data['random_input_len'].values.reshape(-1,1)
    y = data['mean_ttft_ms'].values

    reg = LinearRegression().fit(x, y)

    print(f'    Fixed OSL: {osl}')
    print(f"    Prefill rate: {reg.coef_[0]:.4f} ms/token\n")

seqlen_df = pd.read_csv("./results/seqlen_df")

# prefill analysis
for gpu_type in ['a100', 'v100']:
    print(f"{gpu_type.upper()} ESTIMATES")
    for osl in [128,256,512,1024,2048]:
        compute_prefill_rate(seqlen_df, gpu_type=gpu_type, osl=osl)
