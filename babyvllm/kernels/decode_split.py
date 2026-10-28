"""Sequence splits for paged decode attention (flash-decoding).

The single-pass decode kernel runs one CTA per (sequence, KV head). With four
KV heads and a small batch, most SMs sit idle while each CTA walks the whole
context. Splitting the context into fixed-size partitions gives every
(sequence, KV head) one CTA per partition, and a second kernel reweights the
partial outputs by their log-sum-exp.

Pure Python so the plan is testable without a GPU.
"""

import os

# "auto" splits only when batch × KV heads leaves SMs idle, with
# auto_partition_blocks(batch) pages per split. "0" is always the single-pass
# kernel. A positive integer always splits, with that many pages per split.
# Kept switchable for A/B runs.
DECODE_PARTITION_ENV = "BABYVLLM_DECODE_PARTITION"
# Below 4 pages per split, the partial writes and the reduce cost more than
# the extra CTAs return (H100, Qwen2.5-7B, context 2048).
MIN_PARTITION_BLOCKS = 4


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _next_pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


def auto_partition_blocks(batch: int) -> int:
    """Pages per split. Grows with the batch so a 2048-token step launches a few hundred CTAs.

    Depends only on the batch rounded up to a power of two, never on the
    context. The CUDA graph batch buckets are powers of two, so eager decode
    at batch 5 and the batch-8 graph cut a row at the same pages and return
    the same bits. A row that fits in one partition matches the single-pass
    kernel exactly.
    """
    return max(MIN_PARTITION_BLOCKS, _next_pow2(batch))


def grid_fills_gpu(batch: int, n_kv_heads: int, num_sms: int) -> bool:
    return _next_pow2(batch) * n_kv_heads >= num_sms


def decode_split_plan(
    batch: int,
    n_kv_heads: int,
    block_size: int,
    max_blocks: int,
    max_seq_len: int | None,
    num_sms: int,
    partition_blocks: int | None = None,
) -> tuple[int, int]:
    """(num_splits, blocks_per_split) for one decode launch.

    ``max_seq_len`` is the context bucket, not the live lengths: under CUDA
    graphs the grid is frozen at capture. ``partition_blocks`` overrides the
    env switch, with the same meaning. One split is the single-pass kernel.
    """
    ctx_blocks = max_blocks if max_seq_len is None else _cdiv(max_seq_len, block_size)
    ctx_blocks = max(1, min(ctx_blocks, max_blocks))
    if partition_blocks is None:
        env = os.environ.get(DECODE_PARTITION_ENV, "auto")
        if env == "auto":
            if grid_fills_gpu(batch, n_kv_heads, num_sms):
                return 1, ctx_blocks
            partition_blocks = auto_partition_blocks(batch)
        else:
            partition_blocks = int(env)
    if partition_blocks < 0:
        raise ValueError(
            f"{DECODE_PARTITION_ENV} must be 'auto', 0 or a page count, got {partition_blocks}"
        )
    if partition_blocks == 0:
        return 1, ctx_blocks
    return _cdiv(ctx_blocks, partition_blocks), partition_blocks
