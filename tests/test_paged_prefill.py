import pytest
import torch

try:
    from babyvllm.kernels.paged_prefill import paged_prefill_attention
except ImportError:
    paged_prefill_attention = None

import babyvllm.layers.attention as attn_mod
from babyvllm.layers.attention import Attention, _paged_attention_torch
from babyvllm.worker.context import AttentionMetadata, set_forward_context

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or paged_prefill_attention is None,
        reason="Requires CUDA and Triton",
    ),
]

PAGE = 16


def build_case(specs, num_kv_heads, q_per_kv, head_dim, seed=0):
    """specs: list of (context_len, query_len). Unwritten cache slots are NaN."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_heads = num_kv_heads * q_per_kv
    seq_lens = [c + q for c, q in specs]
    blocks_per_seq = [(s + PAGE - 1) // PAGE for s in seq_lens]
    total_blocks = sum(blocks_per_seq) + 3
    max_blocks = max(blocks_per_seq)

    kv_cache = torch.full(
        (2, total_blocks, PAGE, num_kv_heads, head_dim), float("nan"),
        device="cuda", dtype=torch.bfloat16,
    )
    perm = torch.randperm(total_blocks, device="cuda", generator=g).tolist()
    tables = []
    written = []
    for n, seq_len in zip(blocks_per_seq, seq_lens):
        table = [perm.pop() for _ in range(n)]
        written.extend(table[pos // PAGE] * PAGE + pos % PAGE for pos in range(seq_len))
        tables.append(table + [0] * (max_blocks - n))
    flat = kv_cache.view(2, -1, num_kv_heads, head_dim)
    flat[:, torch.tensor(written, device="cuda")] = torch.randn(
        2, len(written), num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g,
    )

    q_lens = [q for _, q in specs]
    q = torch.randn(sum(q_lens), num_heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    qsl = [0]
    for n in q_lens:
        qsl.append(qsl[-1] + n)
    ctx = [c for c, _ in specs]
    md = AttentionMetadata(
        slot_mapping=torch.zeros(sum(q_lens), dtype=torch.int64, device="cuda"),
        block_tables=torch.tensor(tables, dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        context_lens=torch.tensor(ctx, dtype=torch.int32, device="cuda"),
        query_start_loc_cpu=qsl,
        seq_lens_cpu=seq_lens,
        context_lens_cpu=ctx,
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
    )
    return q, kv_cache, md


def reference(q, kv_cache, md, q_per_kv):
    scale = q.shape[-1] ** -0.5
    return _paged_attention_torch(q.float(), kv_cache.float(), md, scale, q_per_kv)


CASES = {
    "fresh_prefill": [(0, 128)],
    "fresh_prefill_ragged": [(0, 1), (0, 17), (0, 100), (0, 64)],
    "prefix_hit_aligned": [(32, 5), (48, 77), (16, 2)],
    "prefix_unaligned": [(37, 20), (5, 3), (100, 130)],
    "mixed_prefill_and_decode": [(0, 40), (200, 1), (63, 1), (500, 1)],
    "long_tail_after_prefix": [(1024, 1024)],
    "batch64_fresh": [(0, 128)] * 64,
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("num_kv_heads,q_per_kv,head_dim", [(4, 7, 128), (2, 7, 64), (4, 1, 64)])
def test_paged_prefill_matches_torch(case, num_kv_heads, q_per_kv, head_dim):
    q, kv_cache, md = build_case(CASES[case], num_kv_heads, q_per_kv, head_dim)
    expected = reference(q, kv_cache, md, q_per_kv)
    actual = paged_prefill_attention(q, kv_cache, md, head_dim ** -0.5)
    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


def test_non_contiguous_query_view():
    num_kv_heads, q_per_kv, head_dim = 4, 7, 128
    q, kv_cache, md = build_case([(16, 33), (0, 70)], num_kv_heads, q_per_kv, head_dim)
    num_heads = num_kv_heads * q_per_kv
    # Same layout as Qwen2Attention: q is a view into the fused QKV output.
    qkv = torch.randn(q.shape[0], (num_heads + 2 * num_kv_heads) * head_dim, device="cuda", dtype=q.dtype)
    view = qkv[:, : num_heads * head_dim].view(-1, num_heads, head_dim)
    view.copy_(q)
    assert not view.is_contiguous()
    expected = reference(q, kv_cache, md, q_per_kv)
    actual = paged_prefill_attention(view, kv_cache, md, head_dim ** -0.5)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


def test_attention_routes_multi_token_cuda_steps_to_triton(monkeypatch):
    num_kv_heads, q_per_kv, head_dim = 4, 7, 128
    q, kv_cache, md = build_case([(32, 9), (0, 20)], num_kv_heads, q_per_kv, head_dim)
    attn = Attention(num_kv_heads * q_per_kv, head_dim, num_kv_heads)
    attn.kv_cache = kv_cache
    k = torch.empty(q.shape[0], num_kv_heads, head_dim, device="cuda", dtype=q.dtype)

    calls = []
    import babyvllm.kernels.paged_prefill as pp
    real = pp.paged_prefill_attention
    monkeypatch.setattr(pp, "paged_prefill_attention", lambda *a: calls.append(1) or real(*a))
    monkeypatch.setattr(attn_mod, "_PREFILL_BACKEND", "triton")
    with set_forward_context(md):
        actual = attn(q, k, k)
    assert calls == [1]
    torch.testing.assert_close(actual.float(), reference(q, kv_cache, md, q_per_kv), atol=2e-2, rtol=2e-2)

    monkeypatch.setattr(attn_mod, "_PREFILL_BACKEND", "torch")
    with set_forward_context(md):
        attn(q, k, k)
    assert calls == [1]
