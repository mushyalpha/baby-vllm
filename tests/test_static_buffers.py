import torch
import pytest
from babyvllm.worker.model_runner import ModelRunner
from babyvllm.config import ModelConfig, CacheConfig

def test_static_buffers_init():
    cfg = ModelConfig(
        vocab_size=100,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    runner = ModelRunner(cfg, cache_config=CacheConfig(), device="cpu")
    # This should not raise TypeError
    buf = runner._ensure_static_buffers(batch_bucket=4, ctx_bucket=256)
    assert buf is not None
