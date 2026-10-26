import importlib.util
import sys
from pathlib import Path

import pytest
import torch

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"


def _load():
    sys.path.insert(0, str(BENCH_DIR))
    path = BENCH_DIR / "benchmark_prefix_caching.py"
    spec = importlib.util.spec_from_file_location("benchmark_prefix_caching", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prompts_share_exactly_the_system_prompt():
    mod = _load()
    base = [20, 21, 22, 23]
    prompts = mod.make_shared_prefix_prompts(base, n=40, system_len=96, suffix_len=16, vocab_size=151936)
    assert {len(p) for p in prompts} == {112}
    assert len({tuple(p[:96]) for p in prompts}) == 1
    assert len({tuple(p[96:]) for p in prompts}) == 40


def test_salt_changes_the_first_token_so_no_block_is_shared():
    mod = _load()
    a = mod.make_shared_prefix_prompts([20, 21], 1, 64, 8, 151936, salt=0)[0]
    b = mod.make_shared_prefix_prompts([20, 21], 1, 64, 8, 151936, salt=1)[0]
    assert a[0] != b[0]
    assert a[1:64] == b[1:64]


def test_cached_tokens_never_include_the_last_prompt_token():
    mod = _load()
    assert mod.cached_tokens_per_hit(1024, 1088, 16) == 1024
    # The prompt is exactly the system prompt: one block must stay uncached.
    assert mod.cached_tokens_per_hit(32, 32, 16) == 16
    # Partial trailing system block is not shared.
    assert mod.cached_tokens_per_hit(40, 104, 16) == 32


def test_expected_hit_rate():
    mod = _load()
    # 16 requests, first cold: 15 * 1024 / (16 * 1088)
    assert mod.expected_hit_rate(1024, 64, 16, 16) == pytest.approx(15 * 1024 / (16 * 1088))
    assert mod.expected_hit_rate(1024, 64, 16, 16, first_is_cold=False) == pytest.approx(1024 / 1088)


def test_percentile_and_ids_match():
    mod = _load()
    assert mod.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert mod.ids_match([[1], [2]], [[1], [2]])["match"]
    bad = mod.ids_match([[1], [2]], [[1], [3]])
    assert not bad["match"] and bad["mismatched_sequences"] == 1


def _tiny_engine():
    from babyvllm.config import CacheConfig, ModelConfig, SchedulerConfig
    from babyvllm.engine import LLMEngine
    from babyvllm.worker.model_runner import ModelRunner

    cfg = ModelConfig(
        vocab_size=2048,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    runner = ModelRunner(cfg, cache_config=CacheConfig(num_cpu_blocks=400), device="cpu")
    with torch.no_grad():
        for p in runner.model.parameters():
            p.normal_(0, 0.05)
    return LLMEngine(runner, max_num_batched_tokens=2048, max_num_seqs=8), cfg


def test_measured_hit_rate_matches_block_arithmetic_on_a_tiny_model():
    mod = _load()
    engine, cfg = _tiny_engine()
    device = torch.device("cpu")
    bs = engine.block_size
    system_len, suffix_len, n = 64, 16, 4
    prompts = mod.make_shared_prefix_prompts([20, 21, 22], n, system_len, suffix_len, cfg.vocab_size)

    mod.set_caching(engine, False)
    off = mod.run_sequential(engine, prompts, device)
    assert off["hit_rate"] == 0.0

    mod.set_caching(engine, True)
    on = mod.run_sequential(engine, prompts, device)
    expected = mod.expected_hit_rate(system_len, suffix_len, bs, n)
    assert on["hit_rate"] == pytest.approx(expected)
    assert on["warm_hit_rate"] == pytest.approx(system_len / (system_len + suffix_len))
    assert on["ids"] == off["ids"]

    mod.set_caching(engine, True)
    burst_on = mod.run_burst(engine, prompts, 4, device)
    mod.set_caching(engine, False)
    burst_off = mod.run_burst(engine, prompts, 4, device)
    # Request 0 fills the blocks and the other three share them in the same step.
    assert burst_on["hit_rate"] == pytest.approx(expected)
    assert burst_on["ids"] == burst_off["ids"]
    assert burst_on["peak_blocks"] < burst_off["peak_blocks"]
    assert burst_on["prefill_tokens_computed"] < burst_off["prefill_tokens_computed"]
