"""Split plan for flash-decoding. Pure Python, runs on CPU."""
import pytest

from babyvllm.kernels.decode_split import (
    DECODE_PARTITION_ENV,
    MIN_PARTITION_BLOCKS,
    auto_partition_blocks,
    decode_split_plan,
    grid_fills_gpu,
)

H100_SMS = 132
BLOCK = 16
# max_blocks_per_seq for the 2048 bucket: 2048 / 16 + 1.
MAX_BLOCKS = 129
GRAPH_BATCHES = [1, 2, 4, 8, 16, 32, 64]


@pytest.fixture(autouse=True)
def _auto(monkeypatch):
    monkeypatch.delenv(DECODE_PARTITION_ENV, raising=False)


def plan(batch, ctx, n_kv=4, sms=H100_SMS, **kw):
    return decode_split_plan(batch, n_kv, BLOCK, MAX_BLOCKS, ctx, sms, **kw)


def test_batch_one_long_context_splits_at_the_floor():
    assert plan(1, 2048) == (2048 // BLOCK // MIN_PARTITION_BLOCKS, MIN_PARTITION_BLOCKS)


@pytest.mark.parametrize("batch, per", [(1, 4), (2, 4), (4, 4), (8, 8), (16, 16), (32, 32)])
def test_partition_grows_with_the_batch_bucket(batch, per):
    splits, got = plan(batch, 2048)
    assert got == per
    assert splits == 128 // per


def test_full_batch_keeps_the_single_pass_kernel():
    # 64 × 4 = 256 CTAs already cover 132 SMs.
    assert plan(64, 2048) == (1, 128)
    assert plan(64, 256) == (1, 16)


def test_threshold_uses_the_batch_bucket():
    assert not grid_fills_gpu(32, 4, H100_SMS)
    # 33 rounds up to the 64 bucket.
    assert grid_fills_gpu(33, 4, H100_SMS)


@pytest.mark.parametrize("bucket", GRAPH_BATCHES)
def test_eager_batch_and_its_graph_bucket_get_the_same_plan(bucket):
    """Eager decode sees the live batch, the graph its bucket. They must cut rows at the same pages."""
    lo = bucket // 2 + 1 if bucket > 1 else 1
    for live in range(lo, bucket + 1):
        assert auto_partition_blocks(live) == auto_partition_blocks(bucket)
        assert grid_fills_gpu(live, 4, H100_SMS) == grid_fills_gpu(bucket, 4, H100_SMS)
        assert grid_fills_gpu(live, 2, H100_SMS) == grid_fills_gpu(bucket, 2, H100_SMS)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("ctx", [17, 128, 129, 256, 2048])
def test_partition_does_not_depend_on_the_context(batch, ctx):
    """Eager decode plans from the live length, the graph from its bucket."""
    splits, per = plan(batch, ctx)
    assert per == auto_partition_blocks(batch)
    ctx_blocks = -(-ctx // BLOCK)
    assert (splits - 1) * per < ctx_blocks <= splits * per


def test_context_within_one_partition_is_one_split():
    assert plan(1, MIN_PARTITION_BLOCKS * BLOCK) == (1, MIN_PARTITION_BLOCKS)


def test_no_bucket_uses_the_block_table_width():
    assert decode_split_plan(1, 4, BLOCK, MAX_BLOCKS, None, H100_SMS) == (
        -(-MAX_BLOCKS // MIN_PARTITION_BLOCKS), MIN_PARTITION_BLOCKS,
    )


def test_bucket_wider_than_table_is_clamped():
    assert plan(1, 8192) == (-(-MAX_BLOCKS // MIN_PARTITION_BLOCKS), MIN_PARTITION_BLOCKS)


def test_env_zero_is_the_single_pass_kernel(monkeypatch):
    monkeypatch.setenv(DECODE_PARTITION_ENV, "0")
    assert plan(1, 2048) == (1, 128)


def test_env_count_forces_splits_even_at_full_batch(monkeypatch):
    monkeypatch.setenv(DECODE_PARTITION_ENV, "32")
    assert plan(64, 2048) == (4, 32)
    monkeypatch.setenv(DECODE_PARTITION_ENV, "8")
    assert plan(1, 2048) == (16, 8)


def test_partition_that_does_not_divide_rounds_up():
    assert plan(1, 2048, partition_blocks=7) == (19, 7)


def test_explicit_partition_overrides_env(monkeypatch):
    monkeypatch.setenv(DECODE_PARTITION_ENV, "0")
    assert plan(1, 2048, partition_blocks=16) == (8, 16)


def test_negative_partition_is_rejected(monkeypatch):
    monkeypatch.setenv(DECODE_PARTITION_ENV, "-3")
    with pytest.raises(ValueError):
        plan(1, 2048)
