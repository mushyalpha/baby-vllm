import torch
import torch.nn.functional as F

from babyvllm.layers.attention import Attention, _naive_causal_attention, store_kvcache
from babyvllm.worker.context import AttentionMetadata, set_forward_context

def test_prefill_parity():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    q_len = 10
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    
    q = torch.randn(q_len, num_heads, head_dim)
    k = torch.randn(q_len, num_kv_heads, head_dim)
    v = torch.randn(q_len, num_kv_heads, head_dim)
    
    expected = attn(q, k, v)
    
    block_size = 4
    num_blocks = (q_len + block_size - 1) // block_size
    kv_cache = torch.empty(2, num_blocks, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    md = AttentionMetadata(
        slot_mapping=torch.arange(q_len),
        block_tables=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens=torch.tensor([q_len], dtype=torch.int32),
        context_lens=torch.tensor([0], dtype=torch.int32),
        query_start_loc_cpu=[0, q_len],
        seq_lens_cpu=[q_len],
        context_lens_cpu=[0],
        max_query_len=q_len,
        max_seq_len=q_len,
    )
    
    with set_forward_context(md):
        actual = attn(q, k, v)
        
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

def test_prefill_then_decode_parity():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    
    q_all = torch.randn(11, num_heads, head_dim)
    k_all = torch.randn(11, num_kv_heads, head_dim)
    v_all = torch.randn(11, num_kv_heads, head_dim)
    
    expected = _naive_causal_attention(q_all, k_all, v_all, attn.scale, attn.num_queries_per_kv)
    
    block_size = 4
    kv_cache = torch.empty(2, 3, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    store_kvcache(k_all[:10], v_all[:10], kv_cache, torch.arange(10))
    
    q_decode = q_all[10:11]
    k_decode = k_all[10:11]
    v_decode = v_all[10:11]
    
    md = AttentionMetadata(
        slot_mapping=torch.tensor([10]),
        block_tables=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([11], dtype=torch.int32),
        context_lens=torch.tensor([10], dtype=torch.int32),
        query_start_loc_cpu=[0, 1],
        seq_lens_cpu=[11],
        context_lens_cpu=[10],
        max_query_len=1,
        max_seq_len=11,
    )
    
    with set_forward_context(md):
        actual_decode = attn(q_decode, k_decode, v_decode)
        
    torch.testing.assert_close(actual_decode[0], expected[10], atol=1e-5, rtol=1e-5)


def test_batched_decode():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 4

    attn = Attention(num_heads, head_dim, num_kv_heads)

    q0 = torch.randn(7, num_heads, head_dim)
    k0 = torch.randn(7, num_kv_heads, head_dim)
    v0 = torch.randn(7, num_kv_heads, head_dim)
    q1 = torch.randn(6, num_heads, head_dim)
    k1 = torch.randn(6, num_kv_heads, head_dim)
    v1 = torch.randn(6, num_kv_heads, head_dim)

    exp0 = _naive_causal_attention(q0, k0, v0, attn.scale, attn.num_queries_per_kv)
    exp1 = _naive_causal_attention(q1, k1, v1, attn.scale, attn.num_queries_per_kv)

    kv_cache = torch.empty(2, 8, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache

    store_kvcache(k0[:6], v0[:6], kv_cache, torch.arange(6))
    store_kvcache(k1[:5], v1[:5], kv_cache, torch.arange(16, 21))

    q = torch.cat([q0[6:7], q1[5:6]], dim=0)
    k = torch.cat([k0[6:7], k1[5:6]], dim=0)
    v = torch.cat([v0[6:7], v1[5:6]], dim=0)

    md = AttentionMetadata(
        slot_mapping=torch.tensor([6, 21]),
        block_tables=torch.tensor([
            [0, 1],
            [4, 5],
        ], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7, 6], dtype=torch.int32),
        context_lens=torch.tensor([6, 5], dtype=torch.int32),
        query_start_loc_cpu=[0, 1, 2],
        seq_lens_cpu=[7, 6],
        context_lens_cpu=[6, 5],
        max_query_len=1,
        max_seq_len=7,
    )

    with set_forward_context(md):
        actual = attn(q, k, v)

    torch.testing.assert_close(actual[0], exp0[6], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1], exp1[5], atol=1e-5, rtol=1e-5)


def test_batched_mixed():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 4
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    kv_cache = torch.empty(2, 6, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    q0 = torch.randn(5, num_heads, head_dim)
    k0 = torch.randn(5, num_kv_heads, head_dim)
    v0 = torch.randn(5, num_kv_heads, head_dim)
    
    k1_ctx = torch.randn(3, num_kv_heads, head_dim)
    v1_ctx = torch.randn(3, num_kv_heads, head_dim)
    store_kvcache(k1_ctx, v1_ctx, kv_cache, torch.arange(8, 11))
    
    q1 = torch.randn(1, num_heads, head_dim)
    k1 = torch.randn(1, num_kv_heads, head_dim)
    v1 = torch.randn(1, num_kv_heads, head_dim)
    
    k2_ctx = torch.randn(8, num_kv_heads, head_dim)
    v2_ctx = torch.randn(8, num_kv_heads, head_dim)
    store_kvcache(k2_ctx, v2_ctx, kv_cache, torch.arange(12, 20))
    
    q2 = torch.randn(1, num_heads, head_dim)
    k2 = torch.randn(1, num_kv_heads, head_dim)
    v2 = torch.randn(1, num_kv_heads, head_dim)
    
    q_batch = torch.cat([q0, q1, q2], dim=0)
    k_batch = torch.cat([k0, k1, k2], dim=0)
    v_batch = torch.cat([v0, v1, v2], dim=0)
    
    md = AttentionMetadata(
        slot_mapping=torch.tensor([0,1,2,3,4, 11, 20]),
        block_tables=torch.tensor([
            [0, 1, -1],
            [2, -1, -1],
            [3, 4, 5]
        ], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5, 6, 7], dtype=torch.int32),
        seq_lens=torch.tensor([5, 4, 9], dtype=torch.int32),
        context_lens=torch.tensor([0, 3, 8], dtype=torch.int32),
        query_start_loc_cpu=[0, 5, 6, 7],
        seq_lens_cpu=[5, 4, 9],
        context_lens_cpu=[0, 3, 8],
        max_query_len=5,
        max_seq_len=9,
    )
    
    with set_forward_context(md):
        actual = attn(q_batch, k_batch, v_batch)
        
    exp0 = _naive_causal_attention(q0, k0, v0, attn.scale, attn.num_queries_per_kv)
    
    exp1 = _naive_causal_attention(
        torch.cat([torch.zeros(3, num_heads, head_dim), q1], dim=0),
        torch.cat([k1_ctx, k1], dim=0),
        torch.cat([v1_ctx, v1], dim=0),
        attn.scale, attn.num_queries_per_kv
    )
    
    exp2 = _naive_causal_attention(
        torch.cat([torch.zeros(8, num_heads, head_dim), q2], dim=0),
        torch.cat([k2_ctx, k2], dim=0),
        torch.cat([v2_ctx, v2], dim=0),
        attn.scale, attn.num_queries_per_kv
    )
    
    torch.testing.assert_close(actual[0:5], exp0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[5:6], exp1[3:4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[6:7], exp2[8:9], atol=1e-5, rtol=1e-5)

def test_non_contiguous():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 2
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    
    q_all = torch.randn(5, num_heads, head_dim)
    k_all = torch.randn(5, num_kv_heads, head_dim)
    v_all = torch.randn(5, num_kv_heads, head_dim)
    
    expected = _naive_causal_attention(q_all, k_all, v_all, attn.scale, attn.num_queries_per_kv)
    
    kv_cache = torch.empty(2, 10, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    slot_mapping = torch.tensor([
        7*2 + 0, 7*2 + 1,
        2*2 + 0, 2*2 + 1,
        5*2 + 0
    ])
    
    md = AttentionMetadata(
        slot_mapping=slot_mapping,
        block_tables=torch.tensor([[7, 2, 5]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        context_lens=torch.tensor([0], dtype=torch.int32),
        query_start_loc_cpu=[0, 5],
        seq_lens_cpu=[5],
        context_lens_cpu=[0],
        max_query_len=5,
        max_seq_len=5,
    )
    
    with set_forward_context(md):
        actual = attn(q_all, k_all, v_all)
        
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_prefill_parity()
    test_prefill_then_decode_parity()
    test_batched_decode()
    test_batched_mixed()
    test_non_contiguous()
    print("All attention tests passed!")
