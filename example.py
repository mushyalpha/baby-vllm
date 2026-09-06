from babyvllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen2-0.5B")
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Baby-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0].text)
