import json
import random

with open("./benchmarking/ShareGPT_V3_unfiltered_cleaned_split.json", encoding="utf-8") as f:
    data = json.load(f)

data = [e for e in data if "conversations" in e and len(e["conversations"]) >= 2]

random.seed(0) # default of vllm is 0
random.shuffle(data)

for entry in data[:10]:
    prompt = entry["conversations"][0]["value"]
    print(repr(prompt[:1000]))
    print()