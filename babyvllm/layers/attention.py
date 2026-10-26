import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from babyvllm.worker.context import get_forward_context

# Multi-token CUDA steps: "triton" is the paged prefill kernel, "torch" is the
# per-sequence SDPA loop below. Kept switchable for A/B runs.
_PREFILL_BACKEND = os.environ.get("BABYVLLM_PREFILL_ATTN", "triton")

# Flash and the memory-efficient backend both require (batch, heads, seq, dim).
# MATH is last so a CPU run, or a shape flash cannot take, still completes.
# A 3D call is (batch, seq, dim) with no head axis, and that dispatch is the
# math backend, which materialises an fp32 (seq, seq) score matrix.
_CAUSAL_SDPA_BACKENDS = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)

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
        if md.max_query_len == 1 and q.is_cuda and self.kv_cache.shape[2] >= 16:
            from babyvllm.kernels.paged_attn import paged_decode_attention
            return paged_decode_attention(q, self.kv_cache, md.block_tables, md.seq_lens, self.scale)
        if md.max_query_len > 1 and q.is_cuda and _PREFILL_BACKEND == "triton":
            from babyvllm.kernels.paged_prefill import paged_prefill_attention
            return paged_prefill_attention(q, self.kv_cache, md, self.scale)
        if md.max_query_len == 1:
            return _static_decode_attention(
                q, self.kv_cache, md, self.scale,
                self.num_heads, self.num_kv_heads, self.head_dim,
                self.num_queries_per_kv,
            )
        return _paged_attention_torch(q, self.kv_cache, md, self.scale,
                                      self.num_queries_per_kv)


def gather_kv_batched(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    ctx_bucket: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_blocks_needed = (ctx_bucket + block_size - 1) // block_size
    n_blocks_needed = min(n_blocks_needed, block_tables.shape[1])
    B = block_tables.shape[0]
    n_kv_heads = kv_cache.shape[3]
    head_dim = kv_cache.shape[4]

    bt = block_tables[:, :n_blocks_needed]
    bt_flat = bt.reshape(-1)

    k_blocks = kv_cache[0][bt_flat]
    v_blocks = kv_cache[1][bt_flat]

    gathered = n_blocks_needed * block_size
    k = k_blocks.reshape(B, gathered, n_kv_heads, head_dim)
    v = v_blocks.reshape(B, gathered, n_kv_heads, head_dim)

    k = k[:, :ctx_bucket].permute(0, 2, 1, 3)
    v = v[:, :ctx_bucket].permute(0, 2, 1, 3)
    return k, v


def make_decode_mask(
    seq_lens: torch.Tensor,
    ctx_bucket: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(ctx_bucket, device=device)
    seq_lens_long = seq_lens.to(torch.int64)
    valid = positions[None, :] < seq_lens_long[:, None]
    additive = torch.zeros(
        seq_lens.shape[0], 1, 1, ctx_bucket, dtype=dtype, device=device
    )
    return additive.masked_fill(~valid[:, None, None, :], float("-inf"))


def _static_decode_attention(
    q, kv_cache, md, scale, num_heads, num_kv_heads, head_dim, num_queries_per_kv,
):
    B = md.seq_lens.shape[0]
    ctx_bucket = md.max_seq_len
    block_size = kv_cache.shape[2]

    K, V = gather_kv_batched(kv_cache, md.block_tables, ctx_bucket, block_size)
    Q = q.reshape(B, num_heads, 1, head_dim)
    mask = make_decode_mask(md.seq_lens, K.shape[2], q.device, q.dtype)

    group = num_queries_per_kv
    if group > 1:
        Q = Q.reshape(B, num_kv_heads, group, head_dim)
    out = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=mask,
        scale=scale,
    )
    if group > 1:
        out = out.reshape(B, num_heads, 1, head_dim)
    return out.squeeze(2).reshape(-1, num_heads, head_dim)


def _paged_attention_torch(query, kv_cache, md, scale, num_queries_per_kv):
    block_size = kv_cache.shape[2]
    num_kv_heads, head_dim = kv_cache.shape[3], kv_cache.shape[4]
    out = torch.empty_like(query)

    for i, (start, end) in enumerate(zip(md.query_start_loc_cpu[:-1],
                                         md.query_start_loc_cpu[1:])):
        seq_len = md.seq_lens_cpu[i]
        ctx_len = md.context_lens_cpu[i]
        q_len = end - start

        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = md.block_tables[i, :num_blocks]

        k_i = kv_cache[0][blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v_i = kv_cache[1][blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        q_i = query[start:end]

        if ctx_len == 0 and q_len == seq_len:
            # Full prefill. q_len == seq_len, so a causal mask is top-left and
            # the 4D flash/mem-efficient path is exact.
            out[start:end] = _causal_sdpa(q_i, k_i, v_i, scale)
        else:
            # Prefix + a shorter query. is_causal is top-left when the lengths
            # differ, which hides the cached prefix. The explicit tril keeps
            # query i attending through key i + ctx_len. A bool mask stays on
            # the math backend; this branch is one token wide in decode.
            if num_queries_per_kv > 1:
                k_i = k_i.repeat_interleave(num_queries_per_kv, dim=1)
                v_i = v_i.repeat_interleave(num_queries_per_kv, dim=1)
            q_i = q_i.transpose(0, 1)
            k_i = k_i.transpose(0, 1)
            v_i = v_i.transpose(0, 1)
            mask = torch.ones(q_len, seq_len, dtype=torch.bool, device=query.device)
            mask = torch.tril(mask, diagonal=ctx_len)
            o = F.scaled_dot_product_attention(q_i, k_i, v_i,
                                               attn_mask=mask, scale=scale)
            out[start:end] = o.transpose(0, 1)
    return out


def _causal_sdpa(q, k, v, scale):
    """Causal attention for (seq, heads, dim) Q/K/V.

    4D ``(1, heads, seq, dim)`` is the layout flash and mem-efficient SDPA
    accept. The previous 3D call was dispatched to the math backend, which
    writes an fp32 score matrix of shape ``(seq, seq)``: 56.25 GiB at
    seq = 122_880.
    """
    q = q.transpose(0, 1).unsqueeze(0).contiguous()
    k = k.transpose(0, 1).unsqueeze(0).contiguous()
    v = v.transpose(0, 1).unsqueeze(0).contiguous()
    with sdpa_kernel(list(_CAUSAL_SDPA_BACKENDS)):
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            scale=scale,
            enable_gqa=q.shape[1] != k.shape[1],
        )
    return out.squeeze(0).transpose(0, 1).contiguous()


def _naive_causal_attention(q, k, v, scale, num_queries_per_kv):
    """Single-sequence, no-cache reference. Also the memory-sizing forward."""
    if q.shape[1] != k.shape[1] * num_queries_per_kv:
        raise ValueError(
            f"query heads {q.shape[1]} != kv heads {k.shape[1]} × {num_queries_per_kv}"
        )
    return _causal_sdpa(q, k, v, scale)
