#%%
import pandas as pd
import matplot2tikz
from utils.plotting import *
from utils.shared_config import *

set_figure_style()

if __name__ == "__main__":
    pass
    #%%
    results_dir = "../results/"

    df = pd.read_csv(f"{results_dir}concurrency-sweep/averaged_concurrency_sweep_results.csv")

    a100_1b_default = build_tp_series('a100', 'llama-3.2-1b')
    a100_1b_p2pdisabled = build_tp_series('a100', 'llama-3.2-1b', 'p2p_disabled')

    a100_8b_default = build_tp_series('v100', 'llama-3.1-8b')
    a100_8b_p2pdisabled = build_tp_series('v100', 'llama-3.1-8b', 'p2p_disabled')

    a100_13b_default = build_tp_series('a100', 'llama-2-13b')
    a100_13b_p2pdisabled = build_tp_series('a100', 'llama-2-13b', 'p2p_disabled')

    slo_cols = {
        'responsive': 'limit_responsive_90', 'acceptable': 'limit_acceptable_90'
    }
    slo_styles = {
        'responsive': ('*', 60, '--'), 'acceptable': ('D', 30, ':'),
    }

    slo_summary = pd.read_csv(f"{results_dir}/concurrency-sweep/slo_attainment.csv")

    fig = plot_metric_vs_concurrency(
        a100_1b_default, df,
        metric="median_ttft_ms",
        ylabel="ITL",
        title="ITL Across TP Configs",
        # log_scale=False,
        slo_df=slo_summary, slo_cols=slo_cols, slo_styles=slo_styles,

    )
    slo_cols = {
        'responsive': 'pergpu_limit_responsive_90', 'acceptable': 'pergpu_limit_acceptable_90'
    }

    fig_a100_thp_normalized = plot_normalized_throughput(
        a100_1b_default, df,
        title="Smaller Model: Normalized Output Throughput",
        slo_df=slo_summary, slo_cols=slo_cols, slo_styles=slo_styles,
    )

    fig_a100_thp_normalized = plot_normalized_throughput(
        a100_1b_p2pdisabled, df,
        title="Smaller Model No P2P: Normalized Output Throughput",
        slo_df=slo_summary, slo_cols=slo_cols, slo_styles=slo_styles,
    )

    fig_a100_thp_normalized = plot_normalized_throughput(
        a100_8b_default, df,
        title="Primary Model: Normalized Output Throughput",
        slo_df=slo_summary, slo_cols=slo_cols, slo_styles=slo_styles,
    )

    fig_a100_thp_normalized = plot_normalized_throughput(
        a100_8b_p2pdisabled, df,
        title="Primary Model No P2P: Normalized Output Throughput",
        slo_df=slo_summary, slo_cols=slo_cols, slo_styles=slo_styles,
    )
    fig_a100_thp_normalized = plot_normalized_throughput(
        a100_13b_default, df,
        title="Larger Model: Normalized Output Throughput",
        slo_df=slo_summary, slo_cols=slo_cols, slo_styles=slo_styles,
    )
    fig_a100_thp_normalized = plot_normalized_throughput(
        a100_13b_p2pdisabled, df,
        title="Larger Model No P2P: Normalized Output Throughput",
        slo_df=slo_summary, slo_cols=slo_cols, slo_styles=slo_styles,
    )

#%%
df = pd.DataFrame(slo_summary)


nvlink_df = df[df['interconnect'] == 'default'].copy()

nvlink_df['gpu_model'] = nvlink_df['gpu_type'].str.upper() + " (" + nvlink_df['model_name'].str.replace('llama-', '') + ")"
summary_df = nvlink_df.rename(columns={'limit_responsive_90': 'capacity'})

pivot = summary_df.pivot(index='tp', columns='gpu_model', values='capacity')

desired_order = [
    'A100 (3.2-1b)', 'A100 (3.1-8b)', 'A100 (2-13b)',
    'V100 (3.2-1b)', 'V100 (3.1-8b)', 'V100 (2-13b)'
]
pivot = pivot.reindex(columns=desired_order)

# plt.figure(figsize=(8, 4), dpi=300)
sns.heatmap(pivot, annot=True, fmt='.0f', cmap="viridis")
plt.title('SLO-Constrained Max Capacity (NVLink Default)')
plt.ylabel('Tensor Parallelism (TP)')
plt.xlabel('GPU & Model Configuration')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

matplot2tikz.save("slo_heatmap.tex")
# %%
