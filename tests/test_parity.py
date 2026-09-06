import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from babyvllm.config import ModelConfig
from babyvllm.models.qwen2 import Qwen2ForCausalLM
from babyvllm.worker.loader import load_model

def test_parity():
    MODEL = "Qwen/Qwen2-0.5B"
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    
    path = snapshot_download(repo_id=MODEL)
    
    cfg = ModelConfig()
    mine = Qwen2ForCausalLM(cfg)
    load_model(mine, path)
    mine.eval()

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0]
    
    with torch.inference_mode():
        ref = hf(ids[None]).logits[0]
        hidden = mine(ids, torch.arange(len(ids)))
        got = mine.compute_logits(hidden)
        
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
    print("Logit parity test passed!")

if __name__ == "__main__":
    test_parity()
