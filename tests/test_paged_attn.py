import pytest
import torch

try:
    from babyvllm.kernels.paged_attn import paged_decode_attention
except ImportError:
    paged_decode_attention = None

from babyvllm.layers.attention import _paged_attention_torch
from babyvllm.worker.context import AttentionMetadata

pytestmark = pytest.mark.gpu

@pytest.mark.skipif(not torch.cuda.is_available() or paged_decode_attention is None, reason="Requires CUDA and Triton")
@pytest.mark.parametrize("B", [1, 2, 7, 64])
@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 31, 32, 127, 128, 2048])
def test_paged_decode_parity(B, seq_len):
    num_kv_heads = 4
    q_per_kv = 7
    num_heads = num_kv_heads * q_per_kv
    head_dim = 128
    block_size = 16
    
    q = torch.randn(B, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    num_blocks = (seq_len + block_size - 1) // block_size
    max_blocks = num_blocks + 2
    
    # Shuffle block table to ensure we aren't relying on contiguous blocks
    total_phys_blocks = B * max_blocks
    kv_cache = torch.randn(2, total_phys_blocks, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    phys_indices = torch.randperm(total_phys_blocks, device="cuda")
    block_tables = phys_indices.view(B, max_blocks).to(torch.int32)
    
    seq_lens = torch.tensor([seq_len] * B, dtype=torch.int32, device="cuda")
    
    scale = head_dim ** -0.5
    
    actual = paged_decode_attention(q, kv_cache, block_tables, seq_lens, scale)
    
    # Run reference (torch paged attn expects metadata)
    md = AttentionMetadata(
        slot_mapping=torch.arange(B),
        block_tables=block_tables,
        query_start_loc=torch.arange(B + 1, dtype=torch.int32, device="cuda"),
        seq_lens=seq_lens,
        context_lens=seq_lens - 1,
        query_start_loc_cpu=list(range(B + 1)),
        seq_lens_cpu=[seq_len] * B,
        context_lens_cpu=[seq_len - 1] * B,
        max_query_len=1,
        max_seq_len=seq_len,
    )
    
    expected = _paged_attention_torch(q, kv_cache, md, scale, q_per_kv)
    
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
