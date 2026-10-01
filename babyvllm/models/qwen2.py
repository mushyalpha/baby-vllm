import torch
import torch.nn as nn
import torch.nn.functional as F

from babyvllm.layers.layernorm import RMSNorm
from babyvllm.layers.rotary import RotaryEmbedding, apply_rope
from babyvllm.layers.attention import Attention

class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen2Attention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads)

    def forward(self, x, cos, sin):
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        
        q, k = apply_rope(q, k, cos, sin)
        
        o = self.attn(q, k, v)
        return self.o_proj(o.reshape(-1, self.num_heads * self.head_dim))

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(cfg, layer_idx)
        self.mlp = Qwen2MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, hidden_states, residual, cos, sin):
        # 1. input norm & add
        normed, residual = self.input_layernorm(hidden_states, residual)
        
        # 2. attention
        hidden_states = self.self_attn(normed, cos, sin)
        
        # 3. post-attn norm & add
        normed, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # 4. mlp
        hidden_states = self.mlp(normed)
        
        return hidden_states, residual

class Qwen2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            cfg.head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            base=cfg.rope_theta
        )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions, hidden_states.dtype)
        
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, cos, sin)
            
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = Qwen2Model(cfg)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def compute_logits(self, hidden):
        w = self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, w)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)
