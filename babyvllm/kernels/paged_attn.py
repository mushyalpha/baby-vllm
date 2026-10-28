import functools

import torch
import triton
import triton.language as tl

from babyvllm.kernels.decode_split import decode_split_plan

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
        mask = token[None, :] < seq_len
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


@triton.jit
def paged_decode_attn_split_kernel(
    q_ptr, k_cache, v_cache, part_out, part_lse,
    block_tables, seq_lens, scale, num_splits, blocks_per_split,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ps, stride_ph, stride_pn, stride_pd,
    stride_ls, stride_lh, stride_ln,
    stride_bts, stride_btb, stride_sl,
    Q_PER_KV: tl.constexpr, Q_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    # One CTA per (sequence, KV head, split). A split is blocks_per_split pages
    # of the sequence. It writes its softmax-normalised output and its
    # log-sum-exp; the reduce kernel reweights the splits into the final row.
    s = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)

    seq_len = tl.load(seq_lens + s * stride_sl)

    offs_q = tl.arange(0, Q_BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_q < Q_PER_KV
    heads = kvh * Q_PER_KV + offs_q

    q = tl.load(
        q_ptr
        + s * stride_qs
        + heads[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )

    m = tl.full([Q_BLOCK], float("-inf"), dtype=tl.float32)
    l = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, HEAD_DIM], dtype=tl.float32)

    n_blocks = tl.minimum(tl.cdiv(seq_len, BLOCK), MAX_BLOCKS)
    start = split * blocks_per_split
    # The split count comes from the graph's context bucket, not the live
    # lengths. The last split runs to the end of the sequence so a row longer
    # than the bucket is still attended in full.
    end = tl.where(split == num_splits - 1, n_blocks,
                   tl.minimum(start + blocks_per_split, n_blocks))

    bt_ptr = block_tables + s * stride_bts
    phys = tl.load(bt_ptr + start * stride_btb, mask=start < n_blocks, other=0)

    offs_n = tl.arange(0, BLOCK)

    for b in range(start, end):
        next_phys = tl.load(bt_ptr + (b + 1) * stride_btb, mask=(b + 1) < end, other=0)

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

        scores = tl.dot(q, tl.trans(k)) * scale

        token = b * BLOCK + offs_n
        mask = token[None, :] < seq_len
        scores = tl.where(mask, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        l = l * alpha + tl.sum(p, axis=1)
        m = m_new

        phys = next_phys

    # An empty split (past seq_len, or a padded row) stores zeros and -inf,
    # so the reduce gives it zero weight without reading uninitialised memory.
    empty = l == 0
    l_safe = tl.where(empty, 1.0, l)
    o = acc / l_safe[:, None]
    lse = tl.where(empty, float("-inf"), m + tl.log(l_safe))

    tl.store(
        part_out
        + s * stride_ps
        + heads[:, None] * stride_ph
        + split * stride_pn
        + offs_d[None, :] * stride_pd,
        o,
        mask=q_mask[:, None],
    )
    tl.store(
        part_lse + s * stride_ls + heads * stride_lh + split * stride_ln,
        lse,
        mask=q_mask,
    )


@triton.jit
def paged_decode_attn_reduce_kernel(
    part_out, part_lse, out_ptr, num_splits,
    stride_ps, stride_ph, stride_pn, stride_pd,
    stride_ls, stride_lh, stride_ln,
    stride_os, stride_oh, stride_od,
    HEAD_DIM: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)
    lse_row = part_lse + s * stride_ls + h * stride_lh
    out_row = part_out + s * stride_ps + h * stride_ph

    lse_max = tl.load(lse_row)
    for n in range(1, num_splits):
        lse_max = tl.maximum(lse_max, tl.load(lse_row + n * stride_ln))
    # Every split of a padded row is empty; its max is -inf and the row is zero.
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)

    # Accumulate in split order. Empty trailing splits add exact zeros, so a
    # launch with more of them (the graph's wider bucket) gives the same bits.
    num = tl.zeros([HEAD_DIM], dtype=tl.float32)
    den = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for n in range(0, num_splits):
        w = tl.exp(tl.load(lse_row + n * stride_ln) - lse_max)
        o = tl.load(out_row + n * stride_pn + offs_d * stride_pd)
        num += w * o
        den += w
    out = num / tl.where(den == 0, 1.0, den)

    tl.store(
        out_ptr + s * stride_os + h * stride_oh + offs_d * stride_od,
        out.to(out_ptr.dtype.element_ty),
    )


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def paged_decode_attention(q, kv_cache, block_tables, seq_lens, scale,
                           max_seq_len=None, partition_blocks=None):
    """One query token per sequence against its paged K/V.

    ``max_seq_len`` is the context bucket the step was sized for (the graph's
    bucket under CUDA graphs). It sets the split count and must not depend on
    the live lengths, because the grid is frozen at capture.
    ``partition_blocks`` overrides BABYVLLM_DECODE_PARTITION.
    """
    B, n_heads, head_dim = q.shape
    n_kv_heads = kv_cache.shape[3]
    block_size = kv_cache.shape[2]
    
    q_per_kv = n_heads // n_kv_heads
    
    # Pad Q_BLOCK to 16 for Tensor Cores (tl.dot requires >= 16)
    q_block = 16 if q_per_kv <= 16 else triton.next_power_of_2(q_per_kv)

    k_cache = kv_cache[0]
    v_cache = kv_cache[1]

    num_splits, blocks_per_split = decode_split_plan(
        B, n_kv_heads, block_size, block_tables.shape[1], max_seq_len,
        _num_sms(q.device.index if q.device.index is not None else torch.cuda.current_device()),
        partition_blocks,
    )
    if num_splits > 1:
        return _split_decode_attention(
            q, k_cache, v_cache, block_tables, seq_lens, scale,
            num_splits, blocks_per_split, q_per_kv, q_block,
        )

    out = torch.zeros_like(q)

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


def _split_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, scale,
                            num_splits, blocks_per_split, q_per_kv, q_block):
    B, n_heads, head_dim = q.shape
    n_kv_heads = k_cache.shape[2]
    part_out = torch.empty(B, n_heads, num_splits, head_dim, device=q.device, dtype=torch.float32)
    part_lse = torch.empty(B, n_heads, num_splits, device=q.device, dtype=torch.float32)
    # The reduce writes every row, padded ones as zeros, so no fill is needed.
    out = torch.empty_like(q)

    paged_decode_attn_split_kernel[(B, n_kv_heads, num_splits)](
        q, k_cache, v_cache, part_out, part_lse,
        block_tables, seq_lens, float(scale), num_splits, blocks_per_split,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        part_out.stride(0), part_out.stride(1), part_out.stride(2), part_out.stride(3),
        part_lse.stride(0), part_lse.stride(1), part_lse.stride(2),
        block_tables.stride(0), block_tables.stride(1), seq_lens.stride(0),
        Q_PER_KV=q_per_kv,
        Q_BLOCK=q_block,
        HEAD_DIM=head_dim,
        BLOCK=k_cache.shape[1],
        MAX_BLOCKS=block_tables.shape[1],
        num_warps=4,
    )
    paged_decode_attn_reduce_kernel[(B, n_heads)](
        part_out, part_lse, out, num_splits,
        part_out.stride(0), part_out.stride(1), part_out.stride(2), part_out.stride(3),
        part_lse.stride(0), part_lse.stride(1), part_lse.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return out
