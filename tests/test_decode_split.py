"""Split plan for flash-decoding. Pure Python, runs on CPU."""
import pytest

from babyvllm.kernels.decode_split import (
    DECODE_PARTITION_ENV,
    DEFAULT_PARTITION_BLOCKS,
    decode_split_plan,
    grid_fills_gpu,
)

H100_SMS = 132
BLOCK = 16
# max_blocks_per_seq for the 2048 bucket: 2048 / 16 + 1.
MAX_BLOCKS = 129


@pytest.fixture(autouse=True)
def _auto(monkeypatch):
    monkeypatch.delenv(DECODE_PARTITION_ENV, raising=False)


def plan(batch, ctx, n_kv=4, sms=H100_SMS, **kw):
    return decode_split_plan(batch, n_kv, BLOCK, MAX_BLOCKS, ctx, sms, **kw)


def test_batch_one_long_context_splits():
    assert plan(1, 2048) == (2048 // BLOCK // DEFAULT_PARTITION_BLOCKS, DEFAULT_PARTITION_BLOCKS)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32])
def test_batches_that_leave_sms_idle_split(batch):
    splits, per = plan(batch, 2048)
    assert splits > 1 and per == DEFAULT_PARTITION_BLOCKS


def test_full_batch_keeps_the_single_pass_kernel():
    # 64 × 4 = 256 CTAs already cover 132 SMs.
    assert plan(64, 2048) == (1, 128)
    assert plan(64, 256) == (1, 16)


def test_threshold_is_one_cta_per_sm():
    assert not grid_fills_gpu(32, 4, H100_SMS)
    assert grid_fills_gpu(33, 4, H100_SMS)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("ctx", [17, 128, 129, 256, 2048])
def test_partition_does_not_depend_on_batch_or_bucket(batch, ctx):
    """Eager decode (live lengths) and the graph (bucket) must cut rows at the same pages."""
    splits, per = plan(batch, ctx)
    assert per == DEFAULT_PARTITION_BLOCKS or splits == 1
    ctx_blocks = -(-ctx // BLOCK)
    assert (splits - 1) * per < ctx_blocks <= splits * per


def test_context_within_one_partition_is_one_split():
    assert plan(1, DEFAULT_PARTITION_BLOCKS * BLOCK) == (1, DEFAULT_PARTITION_BLOCKS)


def test_no_bucket_uses_the_block_table_width():
    assert decode_split_plan(1, 4, BLOCK, MAX_BLOCKS, None, H100_SMS) == (
        -(-MAX_BLOCKS // DEFAULT_PARTITION_BLOCKS), DEFAULT_PARTITION_BLOCKS,
    )


def test_bucket_wider_than_table_is_clamped():
    assert plan(1, 8192) == (-(-MAX_BLOCKS // DEFAULT_PARTITION_BLOCKS), DEFAULT_PARTITION_BLOCKS)


def test_env_zero_is_the_single_pass_kernel(monkeypatch):
    monkeypatch.setenv(DECODE_PARTITION_ENV, "0")
    assert plan(1, 2048) == (1, 128)


def test_env_count_forces_splits_even_at_full_batch(monkeypatch):
    monkeypatch.setenv(DECODE_PARTITION_ENV, "32")
    assert plan(64, 2048) == (4, 32)
    monkeypatch.setenv(DECODE_PARTITION_ENV, "4")
    assert plan(1, 2048) == (32, 4)


def test_partition_that_does_not_divide_rounds_up():
    assert plan(1, 2048, partition_blocks=7) == (19, 7)


def test_explicit_partition_overrides_env(monkeypatch):
    monkeypatch.setenv(DECODE_PARTITION_ENV, "0")
    assert plan(1, 2048, partition_blocks=16) == (8, 16)


def test_negative_partition_is_rejected(monkeypatch):
    monkeypatch.setenv(DECODE_PARTITION_ENV, "-3")
    with pytest.raises(ValueError):
        plan(1, 2048)
