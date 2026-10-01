import torch
from babyvllm.models.qwen2 import Qwen2Model

class DummyCfg:
    vocab_size = 100
    hidden_size = 64
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 4
    head_dim = 16
    intermediate_size = 128
    rms_norm_eps = 1e-6
    max_position_embeddings = 128
    rope_theta = 10000.0
    tie_word_embeddings = False

cfg = DummyCfg()
model = Qwen2Model(cfg)
input_ids = torch.randint(0, 100, (1, 10))
positions = torch.arange(10).unsqueeze(0)
out = model(input_ids, positions)
print("CPU Model Output Shape:", out.shape)
print("CPU Model smoke test passed!")
