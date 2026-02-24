import pandas as pd

# load CSV
df = pd.read_csv("./results/single-gpu/serve_results.csv")

# move last 3 columns to the front
cols = list(df.columns)
new_order = cols[-3:] + cols[:-3]
df = df[new_order]

# save CSV
df.to_csv("./results/single-gpu/serve_results.csv", index=False)