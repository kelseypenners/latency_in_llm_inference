from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm import LLM

model_id = "./Llama-3.2-1B"
# llm = LLM(model_id, tensor_parallel_size=1)
# output = llm.generate("san francisco is a")


# load model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# generate
prompt = "The meaning of life is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))