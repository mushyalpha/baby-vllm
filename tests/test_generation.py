import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from babyvllm.llm import LLM
from babyvllm.sequence import SamplingParams

def test_e2e_generation():
    MODEL = "Qwen/Qwen2-0.5B"
    prompts = [
        "The capital of France is",
        "A recipe for chocolate cake:"
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    hf_outputs = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out_ids = hf_model.generate(ids, max_new_tokens=10, do_sample=False)
        out_ids = out_ids[0][ids.shape[1]:]
        hf_outputs.append(out_ids.tolist())
        
    my_llm = LLM(MODEL)
    params = SamplingParams(temperature=0.0, max_tokens=10)
    my_outputs = my_llm.generate(prompts, sampling_params=params)
    
    my_outputs_sorted = sorted(my_outputs, key=lambda x: x.seq_id)
    
    for i in range(len(prompts)):
        print(f"--- Prompt: {prompts[i]} ---")
        print(f"HF Output: {hf_outputs[i]}")
        print(f"My Output: {my_outputs_sorted[i].new_token_ids}")
        assert hf_outputs[i] == my_outputs_sorted[i].new_token_ids
        
    print("End-to-end greedy generation test passed!")

if __name__ == "__main__":
    test_e2e_generation()
