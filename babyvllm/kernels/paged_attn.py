import torch
import triton
import triton.language as tl

@triton.jit
def paged_decode_attn_kernel(
    q_ptr, k_cache, v_cache, out_ptr,
    block_tables, seq_lens, scale,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_os, stride_oh, stride_od,
    stride_bts, stride_btb, stride_sl,
    Q_PER_KV: tl.constexpr, Q_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    s = tl.program_id(0)
    kvh = tl.program_id(1)
    
    seq_len = tl.load(seq_lens + s * stride_sl)
    if seq_len == 0:
        return

    offs_q = tl.arange(0, Q_BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_q < Q_PER_KV

    q = tl.load(
        q_ptr
        + s * stride_qs
        + (kvh * Q_PER_KV + offs_q)[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )

    m = tl.full([Q_BLOCK], float("-inf"), dtype=tl.float32)
    l = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, HEAD_DIM], dtype=tl.float32)

    n_blocks = tl.minimum(tl.cdiv(seq_len, BLOCK), MAX_BLOCKS)
    
    bt_ptr = block_tables + s * stride_bts
    phys = tl.load(bt_ptr)
    
    offs_n = tl.arange(0, BLOCK)
    
    for b in range(0, n_blocks):
        # Load next phys early to hide latency (software pipelining block table)
        next_phys = tl.load(bt_ptr + (b + 1) * stride_btb, mask=(b + 1) < n_blocks, other=0)
        
        k = tl.load(
            k_cache
            + phys * stride_kb
            + offs_n[:, None] * stride_kt
            + kvh * stride_kh
            + offs_d[None, :] * stride_kd,
        )
        v = tl.load(
            v_cache
            + phys * stride_vb
            + offs_n[:, None] * stride_vt
            + kvh * stride_vh
            + offs_d[None, :] * stride_vd,
        )
        
        # Use tl.dot which runs on Tensor Cores!
        scores = tl.dot(q, tl.trans(k)) * scale
        
        token = b * BLOCK + offs_n
        mask = (token[None, :] < seq_len) & q_mask[:, None]
        scores = tl.where(mask, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        
        # Use tl.dot for the value accumulation as well!
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new
        
        phys = next_phys

    l_safe = tl.where(l == 0, 1.0, l)
    out = acc / l_safe[:, None]
    
    tl.store(
        out_ptr
        + s * stride_os
        + (kvh * Q_PER_KV + offs_q)[:, None] * stride_oh
        + offs_d[None, :] * stride_od,
        out.to(out_ptr.dtype.element_ty),
        mask=q_mask[:, None],
    )


def paged_decode_attention(q, kv_cache, block_tables, seq_lens, scale):
    B, n_heads, head_dim = q.shape
    n_kv_heads = kv_cache.shape[3]
    block_size = kv_cache.shape[2]
    
    q_per_kv = n_heads // n_kv_heads
    
    # Pad Q_BLOCK to 16 for Tensor Cores (tl.dot requires >= 16)
    q_block = 16 if q_per_kv <= 16 else triton.next_power_of_2(q_per_kv)
    
    out = torch.empty_like(q)
    k_cache = kv_cache[0]
    v_cache = kv_cache[1]
    
    grid = (B, n_kv_heads)
    
    paged_decode_attn_kernel[grid](
        q, k_cache, v_cache, out,
        block_tables, seq_lens, float(scale),
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables.stride(0), block_tables.stride(1), seq_lens.stride(0),
        Q_PER_KV=q_per_kv,
        Q_BLOCK=q_block,
        HEAD_DIM=head_dim,
        BLOCK=block_size,
        MAX_BLOCKS=block_tables.shape[1],
        num_warps=4,
    )
    return out
