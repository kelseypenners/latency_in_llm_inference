#%% set available devices
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#%% other imports
import torch
import time
import numpy as np
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector


#%% initialize model, prompts and sampling parameters

# model defintion
model_id = "../Llama-3.2-1B"
llm = LLM(model_id, disable_log_stats=False)

# model prompts
prompts = [
    "Hello, my name is",
    "San Francisco is a",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=128)

#%% warm up generation
# warm up
_ = llm.generate(prompts)

#%% generate texts from prompts
outputs = llm.generate(prompts, sampling_params)

# print outpus
print("-" * 40)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"prompt: {prompt!r}\n generated text: {generated_text!r}")
    print(f"metrics: \n{output.metrics}")
    print("-" * 40)


for metric in llm.get_metrics():
    if isinstance(metric, Gauge):
            print(f"{metric.name} (gauge) = {metric.value}")
    elif isinstance(metric, Counter):
        print(f"{metric.name} (counter) = {metric.value}")
    elif isinstance(metric, Vector):
        print(f"{metric.name} (vector) = {metric.values}")
    elif isinstance(metric, Histogram):
        print(f"{metric.name} (histogram)")
        print(f"    sum = {metric.sum}")
        print(f"    count = {metric.count}")
        for bucket_le, value in metric.buckets.items():
            print(f"    {bucket_le} = {value}")

# %%
for prompt in prompts:
    start_time = time.perf_counter()
    outputs = llm.generate(prompt, sampling_params)
    end_time = time.perf_counter()

    output = outputs[0].outputs[0]
    text = output.text

    print(f"prompt: {prompt!r}")
    print(f"generated text: {text!r}")
    print(f"elapsed time: {end_time - start_time:.2f} s")
    print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("-" * 40 )
# %%
