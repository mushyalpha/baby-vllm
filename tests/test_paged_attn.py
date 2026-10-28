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


NUM_KV_HEADS = 4
Q_PER_KV = 7
HEAD_DIM = 128
BLOCK_SIZE = 16
# Table width of the 2048 graph bucket: 2048 / 16 + 1.
GRAPH_TABLE_WIDTH = 129
needs_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or paged_decode_attention is None,
    reason="Requires CUDA and Triton",
)


def _cdiv(a, b):
    return -(-a // b)


def _make_inputs(lens, width=None, seed=0):
    """Ragged decode batch with a shuffled block table. A length of 0 is a padded row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    B = len(lens)
    width = width or _cdiv(max(max(lens), 1), BLOCK_SIZE) + 2
    total = B * width + 1
    q = torch.randn(B, NUM_KV_HEADS * Q_PER_KV, HEAD_DIM, device="cuda",
                    dtype=torch.bfloat16, generator=g)
    kv_cache = torch.randn(2, total, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device="cuda",
                           dtype=torch.bfloat16, generator=g)
    perm = torch.randperm(total, device="cuda", generator=g)[: B * width]
    block_tables = perm.view(B, width).to(torch.int32)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return q, kv_cache, block_tables, seq_lens


def _reference(q, kv_cache, block_tables, lens, scale):
    """fp32 softmax(QK^T)V per row with grouped-query heads. Padded rows are zero."""
    B, H, D = q.shape
    out = torch.zeros(B, H, D, device=q.device, dtype=torch.float32)
    for i, n in enumerate(lens):
        if n == 0:
            continue
        blocks = block_tables[i, : _cdiv(n, BLOCK_SIZE)].long()
        k = kv_cache[0][blocks].reshape(-1, NUM_KV_HEADS, D)[:n].float()
        v = kv_cache[1][blocks].reshape(-1, NUM_KV_HEADS, D)[:n].float()
        k = k.repeat_interleave(Q_PER_KV, dim=1)
        v = v.repeat_interleave(Q_PER_KV, dim=1)
        p = (torch.einsum("hd,lhd->hl", q[i].float(), k) * scale).softmax(-1)
        out[i] = torch.einsum("hl,lhd->hd", p, v)
    return out


RAGGED = [1, 17, 300, 2048, 0, 64, 1000, 16]


@needs_triton
@pytest.mark.parametrize("partition", [0, 1, 2, 3, 7, 8, 64])
def test_split_decode_matches_reference_on_ragged_batch(partition):
    scale = HEAD_DIM ** -0.5
    q, kv_cache, bt, seq_lens = _make_inputs(RAGGED, width=GRAPH_TABLE_WIDTH)
    actual = paged_decode_attention(q, kv_cache, bt, seq_lens, scale,
                                    max_seq_len=2048, partition_blocks=partition)
    expected = _reference(q, kv_cache, bt, RAGGED, scale)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    assert actual[RAGGED.index(0)].abs().max() == 0


@needs_triton
@pytest.mark.parametrize("n", [1, 15, 16, 17, 127, 128, 129, 255, 256, 1023, 2047, 2048])
def test_auto_split_with_bucket_wider_than_live_length(n, monkeypatch):
    """The graph sizes splits from the 2048 bucket; most splits are empty for short rows."""
    from babyvllm.kernels.decode_split import DECODE_PARTITION_ENV, decode_split_plan

    monkeypatch.delenv(DECODE_PARTITION_ENV, raising=False)
    scale = HEAD_DIM ** -0.5
    q, kv_cache, bt, seq_lens = _make_inputs([n], width=GRAPH_TABLE_WIDTH, seed=n)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    splits, _ = decode_split_plan(1, NUM_KV_HEADS, BLOCK_SIZE, GRAPH_TABLE_WIDTH, 2048, sms)
    assert splits > 1, "batch 1 at the 2048 bucket should take the split path"
    actual = paged_decode_attention(q, kv_cache, bt, seq_lens, scale, max_seq_len=2048)
    expected = _reference(q, kv_cache, bt, [n], scale)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


@needs_triton
def test_live_length_and_graph_bucket_give_identical_bits(monkeypatch):
    """Eager decode plans from the live max length, the graph from its bucket."""
    from babyvllm.kernels.decode_split import DECODE_PARTITION_ENV

    monkeypatch.delenv(DECODE_PARTITION_ENV, raising=False)
    scale = HEAD_DIM ** -0.5
    lens = [300, 17, 129, 0, 5]
    q, kv_cache, bt, seq_lens = _make_inputs(lens, width=GRAPH_TABLE_WIDTH)
    eager = paged_decode_attention(q, kv_cache, bt, seq_lens, scale, max_seq_len=max(lens))
    graph = paged_decode_attention(q, kv_cache, bt, seq_lens, scale, max_seq_len=2048)
    assert torch.equal(eager, graph)


@needs_triton
@pytest.mark.parametrize("n", [1, 17, 64, 127, 128])
def test_row_within_one_partition_matches_single_pass_bits(n):
    from babyvllm.kernels.decode_split import DEFAULT_PARTITION_BLOCKS

    assert n <= DEFAULT_PARTITION_BLOCKS * BLOCK_SIZE
    scale = HEAD_DIM ** -0.5
    q, kv_cache, bt, seq_lens = _make_inputs([n, n], width=GRAPH_TABLE_WIDTH, seed=n)
    single = paged_decode_attention(q, kv_cache, bt, seq_lens, scale,
                                    max_seq_len=2048, partition_blocks=0)
    split = paged_decode_attention(q, kv_cache, bt, seq_lens, scale,
                                   max_seq_len=2048, partition_blocks=DEFAULT_PARTITION_BLOCKS)
    assert torch.equal(single, split)


@needs_triton
def test_row_longer_than_the_bucket_is_attended_in_full():
    scale = HEAD_DIM ** -0.5
    lens = [1000, 40]
    q, kv_cache, bt, seq_lens = _make_inputs(lens)
    actual = paged_decode_attention(q, kv_cache, bt, seq_lens, scale,
                                    max_seq_len=256, partition_blocks=4)
    expected = _reference(q, kv_cache, bt, lens, scale)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


@needs_triton
@pytest.mark.parametrize("partition_env", ["0", "auto"])
def test_decode_attention_replays_in_a_cuda_graph(partition_env, monkeypatch):
    """Grid is frozen at capture; lengths and block tables change between replays."""
    from babyvllm.kernels.decode_split import DECODE_PARTITION_ENV

    monkeypatch.setenv(DECODE_PARTITION_ENV, partition_env)
    scale = HEAD_DIM ** -0.5
    q, kv_cache, bt, seq_lens = _make_inputs([2048, 2048, 0, 0], width=GRAPH_TABLE_WIDTH)
    run = lambda: paged_decode_attention(q, kv_cache, bt, seq_lens, scale, max_seq_len=2048)

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()

    g = torch.Generator(device="cuda").manual_seed(7)
    for lens in ([5, 2048, 0, 0], [1999, 33, 129, 0], [1, 1, 1, 1]):
        seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        bt.copy_(torch.randperm(bt.numel(), device="cuda", generator=g).view_as(bt).to(torch.int32))
        q.copy_(torch.randn(q.shape, device="cuda", dtype=q.dtype, generator=g))
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(q, kv_cache, bt, lens, scale)
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
