#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# load data
sweep_df = pd.read_csv('../results/single-gpu/11-02-2026/sweep_results.csv')

# gpu-specific
a100_data = sweep_df[sweep_df['gpu_type'] == 'a100'].copy()

