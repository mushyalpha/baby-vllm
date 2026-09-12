import torch
import torch.nn as nn
import torch.nn.functional as F

from babyvllm.worker.context import get_forward_context

def store_kvcache(key: torch.Tensor,
                  value: torch.Tensor,
                  kv_cache: torch.Tensor,
                  slot_mapping: torch.Tensor) -> None:
    _, _, _, num_kv_heads, head_dim = kv_cache.shape
    k_flat = kv_cache[0].view(-1, num_kv_heads, head_dim)
    v_flat = kv_cache[1].view(-1, num_kv_heads, head_dim)
    k_flat.index_copy_(0, slot_mapping, key.to(k_flat.dtype))
    v_flat.index_copy_(0, slot_mapping, value.to(v_flat.dtype))


class Attention(nn.Module):
    """Paged attention over a flattened (varlen) token batch."""

    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int,
                 scale: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.kv_cache: torch.Tensor | None = None

    def forward(self, q, k, v):
        ctx = get_forward_context()
        if self.kv_cache is None or ctx is None:
            return _naive_causal_attention(q, k, v, self.scale,
                                           self.num_queries_per_kv)
        md = ctx.attn_metadata
        store_kvcache(k, v, self.kv_cache, md.slot_mapping)
        if md.max_query_len == 1:
            #decode
            if q.is_cuda:
                try:
                    from babyvllm.kernels.paged_attn import paged_decode_attention
                    return paged_decode_attention(
                        q, self.kv_cache, md.block_tables, md.seq_lens, self.scale,
                    )
                except ImportError:
                    pass
            return _paged_attention_torch(q, self.kv_cache, md, self.scale,
                                          self.num_queries_per_kv)
        return _paged_attention_torch(q, self.kv_cache, md, self.scale,
                                      self.num_queries_per_kv)


def _paged_attention_torch(query, kv_cache, md, scale, num_queries_per_kv):
    block_size = kv_cache.shape[2]
    num_kv_heads, head_dim = kv_cache.shape[3], kv_cache.shape[4]
    out = torch.empty_like(query)

    for i, (start, end) in enumerate(zip(md.query_start_loc_cpu[:-1],
                                         md.query_start_loc_cpu[1:])):
        seq_len = md.seq_lens_cpu[i]
        ctx_len = md.context_lens_cpu[i]
        q_len = end - start
        if seq_len == 0 or q_len == 0:
            out[start:end] = 0
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = md.block_tables[i, :num_blocks]

        k_i = kv_cache[0][blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v_i = kv_cache[1][blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        if num_queries_per_kv > 1:
            k_i = k_i.repeat_interleave(num_queries_per_kv, dim=1)
            v_i = v_i.repeat_interleave(num_queries_per_kv, dim=1)

        q_i = query[start:end].transpose(0, 1)
        k_i = k_i.transpose(0, 1)
        v_i = v_i.transpose(0, 1)

        if ctx_len == 0 and q_len == seq_len:
            o = F.scaled_dot_product_attention(q_i, k_i, v_i,
                                               is_causal=True, scale=scale)
        else:
            mask = torch.ones(q_len, seq_len, dtype=torch.bool, device=query.device)
            mask = torch.tril(mask, diagonal=ctx_len)
            o = F.scaled_dot_product_attention(q_i, k_i, v_i,
                                               attn_mask=mask, scale=scale)
        out[start:end] = o.transpose(0, 1)
    return out


def _naive_causal_attention(q, k, v, scale, num_queries_per_kv):
    """Single-sequence, no-cache reference. Only for parity tests."""
    if num_queries_per_kv > 1:
        k = k.repeat_interleave(num_queries_per_kv, dim=1)
        v = v.repeat_interleave(num_queries_per_kv, dim=1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                                       v.transpose(0, 1), is_causal=True, scale=scale)
    return o.transpose(0, 1)
