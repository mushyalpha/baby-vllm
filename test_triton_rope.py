import torch
from babyvllm.layers.rotary import apply_rope
from babyvllm.layers.attention import store_kvcache
from babyvllm.kernels.rope import fused_rope_and_cache

torch.manual_seed(42)

def test_rope_and_cache():
    # Shapes for Qwen2.5-7B
    num_tokens = 17
    num_heads = 28
    num_kv_heads = 4
    head_dim = 128
    
    num_blocks = 10
    block_size = 16
    
    # Inputs
    q = torch.randn(num_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    cos = torch.randn(num_tokens, head_dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(num_tokens, head_dim, device="cuda", dtype=torch.bfloat16)
    
    slot_mapping = torch.randint(0, num_blocks * block_size, (num_tokens,), device="cuda", dtype=torch.int64)
    kv_cache = torch.randn(2, num_blocks, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    # Reference (PyTorch)
    q_ref = q.clone()
    k_ref = k.clone()
    v_ref = v.clone()
    kv_cache_ref = kv_cache.clone()
    
    q_ref_out, k_ref_out = apply_rope(q_ref, k_ref, cos, sin)
    store_kvcache(k_ref_out, v_ref, kv_cache_ref, slot_mapping)
    
    # Triton
    q_triton = q.clone()
    k_triton = k.clone()
    v_triton = v.clone()
    kv_cache_triton = kv_cache.clone()
    
    fused_rope_and_cache(q_triton, k_triton, v_triton, cos, sin, kv_cache_triton, slot_mapping)
    
    # Assertions
    torch.testing.assert_close(q_triton, q_ref_out)
    torch.testing.assert_close(k_triton, k_ref_out)
    torch.testing.assert_close(kv_cache_triton, kv_cache_ref)
    print("PASS: Triton RoPE+KV_Cache perfectly matches PyTorch!")

if __name__ == "__main__":
    test_rope_and_cache()
