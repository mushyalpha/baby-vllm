import os
import torch
from huggingface_hub import snapshot_download
from babyvllm.models.qwen2 import Qwen2ForCausalLM
from babyvllm.worker.loader import load_model

class DummyCfg:
    vocab_size = 151936
    hidden_size = 896
    num_hidden_layers = 24
    num_attention_heads = 14
    num_key_value_heads = 2
    head_dim = 64
    intermediate_size = 4864
    rms_norm_eps = 1e-6
    max_position_embeddings = 128
    rope_theta = 10000.0
    tie_word_embeddings = True

model = Qwen2ForCausalLM(DummyCfg())
print("Downloading 0.5B weights (tiny)...")
path = snapshot_download("Qwen/Qwen2.5-0.5B", allow_patterns="*.safetensors")
print("Loading merged weights...")
load_model(model, path)
print("SUCCESS!")
