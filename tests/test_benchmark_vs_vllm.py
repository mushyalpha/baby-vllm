import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_vs_vllm.py"
    spec = importlib.util.spec_from_file_location("benchmark_vs_vllm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prompts_are_exact_and_do_not_share_a_prefix():
    mod = _load()
    prompts = mod.make_prompt_token_ids([20, 21, 22, 23], n=64, prompt_len=128, vocab_size=151936)
    assert len(prompts) == 64
    assert {len(ids) for ids in prompts} == {128}
    assert len({ids[0] for ids in prompts}) == 64
    assert len({tuple(ids) for ids in prompts}) == 64


def test_prefill_budget_caps_the_64_by_128_prefill():
    mod = _load()
    assert mod.prefill_budget([1, 8, 32, 64], 128) == 8192
    assert mod.prefill_budget([1], 128) == 128
    with pytest.raises(ValueError):
        mod.prefill_budget([1], 8193)


def test_graph_keys_stay_in_the_256_bucket_for_the_short_workload():
    mod = _load()
    keys, eager = mod.graph_keys(
        [1, 8, 32, 64], 128, 64, [1, 2, 4, 8, 16, 32, 64], [256, 2048]
    )
    assert eager == []
    assert set(keys) == {(1, 256), (8, 256), (32, 256), (64, 256)}
    assert keys[0] == (64, 256)


def test_graph_keys_capture_both_context_buckets_when_the_run_crosses_256():
    mod = _load()
    keys, eager = mod.graph_keys([4], 200, 100, [1, 2, 4, 8], [256, 2048])
    assert eager == []
    assert set(keys) == {(4, 256), (4, 2048)}
    assert keys[0] == (4, 2048)


def test_graph_keys_reject_a_sequence_past_the_last_bucket():
    mod = _load()
    with pytest.raises(ValueError):
        mod.graph_keys([1], 2048, 64, [1], [256, 2048])


def test_decode_rate_requires_one_token_per_request_at_the_prefill_boundary():
    mod = _load()
    assert mod.decode_tok_s(64 * 64, 64, 1.0, 0.2, 64) == pytest.approx(64 * 63 / 0.8)
    assert mod.decode_tok_s(100, 64, 1.0, 0.2, 80) is None


def test_vllm_kwargs_keep_prefix_caching_off_when_init_forwards_kwargs():
    mod = _load()
    kwargs = mod.vllm_engine_kwargs(
        {"model", "dtype", "kwargs"},
        model="Qwen/Qwen2.5-7B",
        dtype="bfloat16",
        max_batch=64,
        max_model_len=192,
        batched_tokens=8192,
        forwards_kwargs=True,
    )
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enforce_eager"] is False
    assert kwargs["enable_chunked_prefill"] is False
    assert kwargs["gpu_memory_utilization"] == 0.9
    assert kwargs["max_num_batched_tokens"] > kwargs["max_model_len"]
    assert kwargs["max_num_seqs"] == 64


def test_vllm_kwargs_refuse_to_drop_prefix_caching():
    mod = _load()
    with pytest.raises(RuntimeError, match="enable_prefix_caching"):
        mod.vllm_engine_kwargs(
            {"model", "dtype", "enforce_eager", "gpu_memory_utilization"},
            model="Qwen/Qwen2.5-7B",
            dtype="bfloat16",
            max_batch=1,
            max_model_len=192,
            batched_tokens=128,
            forwards_kwargs=False,
        )


def test_publishable_is_only_the_h100_7b_bf16_ladder():
    mod = _load()
    assert mod.publishable("NVIDIA H100 80GB HBM3", "Qwen/Qwen2.5-7B", "bfloat16")
    assert not mod.publishable("NVIDIA GeForce RTX 4090", "Qwen/Qwen2.5-7B", "bfloat16")
    assert not mod.publishable("NVIDIA H100 80GB HBM3", "Qwen/Qwen2.5-0.5B", "bfloat16")
    assert not mod.publishable("NVIDIA H100 80GB HBM3", "Qwen/Qwen2.5-7B", "float16")
