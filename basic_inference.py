import torch
from vllm import LLM, SamplingParams

model_id = "./Llama-3.2-1B"
llm = LLM(model_id)

# model prompts
prompts = [
    "Hello, my name is",
    "San Francisco is a",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"prompt: {prompt!r}, generated text: {generated_text!r}")