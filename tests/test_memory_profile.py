from babyvllm.config import CacheConfig, ModelConfig, SchedulerConfig
from babyvllm.worker.model_runner import (
    MEMORY_PROFILE_TOKENS,
    ModelRunner,
    profile_forward_tokens,
)


def test_b64_budget_is_not_the_profile_sequence_length():
    # Day 6 shape. One causal sequence of this length makes math SDPA
    # allocate a 122_880² fp32 score matrix, which is 56.25 GiB.
    budget = 64 * 1920
    n = profile_forward_tokens(budget, max_position_embeddings=32_768)
    assert budget == 122_880
    assert n == MEMORY_PROFILE_TOKENS == 128
    assert n * n * 4 < 8 * 1024 * 1024


def test_profile_respects_shorter_limits():
    assert profile_forward_tokens(8, 256) == 8
    assert profile_forward_tokens(10_000, 32) == 32
    assert profile_forward_tokens(0, 256) == 1


def test_sizing_forward_runs_the_capped_length_and_one_logit_row():
    cfg = ModelConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    runner = ModelRunner(
        cfg,
        cache_config=CacheConfig(),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=64 * 1920),
        device="cpu",
        use_cuda_graphs=False,
    )
    seen = {}
    orig_forward = runner.model.forward
    orig_logits = runner.model.compute_logits

    def forward(input_ids, positions):
        seen["tokens"] = input_ids.shape[0]
        return orig_forward(input_ids, positions)

    def logits(hidden):
        seen["logit_rows"] = hidden.shape[0]
        return orig_logits(hidden)

    runner.model.forward = forward
    runner.model.compute_logits = logits
    n = profile_forward_tokens(64 * 1920, cfg.max_position_embeddings)
    runner._profile_activation_memory(n)
    assert seen["tokens"] == 128
    assert seen["logit_rows"] == 1
