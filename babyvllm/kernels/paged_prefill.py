import torch
import triton
import triton.language as tl


@triton.jit
def paged_prefill_attn_kernel(
    q_ptr, k_cache, v_cache, out_ptr,
    block_tables, query_start_loc, seq_lens, context_lens, scale,
    stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_bts, stride_btb,
    Q_PER_KV: tl.constexpr, HEAD_DIM: tl.constexpr, PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    m_blk = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    q_start = tl.load(query_start_loc + s)
    q_len = tl.load(query_start_loc + s + 1) - q_start
    if m_blk * BLOCK_M >= q_len:
        return
    seq_len = tl.load(seq_lens + s)
    ctx_len = tl.load(context_lens + s)
    kvh = h // Q_PER_KV

    offs_m = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < q_len

    q = tl.load(
        q_ptr
        + (q_start + offs_m)[:, None] * stride_qt
        + h * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=m_mask[:, None],
        other=0.0,
    )
    # Query row i sits at absolute position ctx_len + i and sees keys 0..ctx_len + i.
    q_pos = ctx_len + offs_m

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Keys past the last query row of this tile are causally invisible to every row in it.
    hi = tl.minimum(seq_len, ctx_len + (m_blk + 1) * BLOCK_M)
    bt_ptr = block_tables + s * stride_bts

    for start_n in range(0, hi, BLOCK_N):
        pos = start_n + offs_n
        n_mask = pos < hi
        phys = tl.load(bt_ptr + (pos // PAGE) * stride_btb, mask=n_mask, other=0).to(tl.int64)
        off = pos % PAGE

        # other=0 matters: slots past seq_len were never written and may hold NaN,
        # and 0 * NaN in the P @ V dot would poison the row.
        k = tl.load(
            k_cache
            + phys[:, None] * stride_kb
            + off[:, None] * stride_kt
            + kvh * stride_kh
            + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_cache
            + phys[:, None] * stride_vb
            + off[:, None] * stride_vt
            + kvh * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * scale
        visible = (pos[None, :] <= q_pos[:, None]) & n_mask[None, :]
        scores = tl.where(visible, scores, float("-inf"))

        # Key 0 is visible to every row on the first tile, so m_new is finite from then on.
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        out_ptr
        + (q_start + offs_m)[:, None] * stride_ot
        + h * stride_oh
        + offs_d[None, :] * stride_od,
        out.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )


def paged_prefill_attention(q, kv_cache, md, scale):
    """Causal attention for a flattened varlen batch, reading K/V from the paged cache.

    Covers a fresh prefill (context 0), a query after a cached prefix, and
    decode rows (query length 1) mixed into the same step. K/V for the query
    tokens must already be in the cache.
    """
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = kv_cache.shape[3]
    page = kv_cache.shape[2]
    assert head_dim & (head_dim - 1) == 0, f"head_dim {head_dim} must be a power of two"
    assert num_heads % num_kv_heads == 0

    out = torch.empty_like(q)
    if num_tokens == 0:
        return out

    k_cache = kv_cache[0]
    v_cache = kv_cache[1]
    num_seqs = md.seq_lens.shape[0]
    # tl.dot needs at least 16 rows. A short-query step (decodes plus a small
    # prefill) would otherwise compute 64 rows per program for one real row.
    block_m = 64 if md.max_query_len >= 64 else 16
    block_n = 64
    grid = (triton.cdiv(md.max_query_len, block_m), num_heads, num_seqs)

    paged_prefill_attn_kernel[grid](
        q, k_cache, v_cache, out,
        md.block_tables, md.query_start_loc, md.seq_lens, md.context_lens, float(scale),
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        md.block_tables.stride(0), md.block_tables.stride(1),
        Q_PER_KV=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        PAGE=page,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return out
