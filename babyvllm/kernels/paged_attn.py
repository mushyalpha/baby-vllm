import torch
import triton
import triton.language as tl


@triton.jit
def paged_decode_attn(
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
    ).to(tl.float32)

    m = tl.full([Q_BLOCK], float("-inf"), dtype=tl.float32)
    l = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, HEAD_DIM], dtype=tl.float32)

    n_blocks = tl.minimum(tl.cdiv(seq_len, BLOCK), MAX_BLOCKS)
    for b in range(0, n_blocks):
        phys = tl.load(block_tables + s * stride_bts + b * stride_btb).to(tl.int64)
        offs_n = tl.arange(0, BLOCK)
        k = tl.load(
            k_cache
            + phys * stride_kb
            + offs_n[:, None] * stride_kt
            + kvh * stride_kh
            + offs_d[None, :] * stride_kd,
        ).to(tl.float32)
        v = tl.load(
            v_cache
            + phys * stride_vb
            + offs_n[:, None] * stride_vt
            + kvh * stride_vh
            + offs_d[None, :] * stride_vd,
        ).to(tl.float32)

        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        token = b * BLOCK + offs_n
        scores = tl.where(
            (token[None, :] < seq_len) & q_mask[:, None],
            scores,
            float("-inf"),
        )

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new

    l_safe = tl.where(l == 0, 1.0, l)
    out = acc / l_safe[:, None]
    out = tl.where(l[:, None] == 0, 0.0, out)

    tl.store(
        out_ptr
        + s * stride_os
        + (kvh * Q_PER_KV + offs_q)[:, None] * stride_oh
        + offs_d[None, :] * stride_od,
        out,
        mask=q_mask[:, None],
    )


def paged_decode_attention(q, kv_cache, block_tables, seq_lens, scale):
    B, n_heads, head_dim = q.shape
    n_kv_heads = kv_cache.shape[3]
    block_size = kv_cache.shape[2]
    q_per_kv = n_heads // n_kv_heads
    q_block = triton.next_power_of_2(q_per_kv)
    out = torch.empty_like(q)
    k_cache = kv_cache[0]
    v_cache = kv_cache[1]
    grid = (B, n_kv_heads)
    paged_decode_attn[grid](
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
    )
    return out
