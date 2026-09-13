"""Qwen2.5-7B bf16 GPU parity. Not part of default CI.

    BABYVLLM_RUN_7B=1 pytest tests/test_parity_7b.py -m gpu
"""
import os

import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from babyvllm.config import PUBLISHED_MODEL
from babyvllm.llm import LLM
from babyvllm.sequence import SamplingParams

pytestmark = pytest.mark.gpu


def _published_path() -> str:
    allow_download = os.environ.get("BABYVLLM_RUN_7B") == "1"
    try:
        return snapshot_download(
            repo_id=PUBLISHED_MODEL, local_files_only=not allow_download
        )
    except Exception as e:
        pytest.skip(
            f"{PUBLISHED_MODEL} not in cache (set BABYVLLM_RUN_7B=1 to download): {e}"
        )


@pytest.fixture(scope="module")
def published_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for 7B tests")
    return _published_path()


@pytest.fixture(scope="module")
def llm_7b(published_path):
    return LLM(published_path, device="cuda", use_cuda_graphs=False)


def test_qwen25_7b_loads_untied_lm_head(llm_7b):
    cfg = llm_7b.model_config
    assert cfg.hidden_size == 3584
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 28
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 128
    assert cfg.vocab_size == 152064
    assert cfg.tie_word_embeddings is False
    assert cfg.num_attention_heads // cfg.num_key_value_heads == 7
    model = llm_7b.model_runner.model
    assert model.lm_head is not None
    assert tuple(model.lm_head.weight.shape) == (152064, 3584)
    caches = [
        m.kv_cache
        for m in model.modules()
        if m.__class__.__name__ == "Attention"
    ]
    assert caches and all(c is not None for c in caches)


def test_parity_7b_logits_and_greedy_ids(llm_7b, published_path):
    device = torch.device("cuda")
    hf = AutoModelForCausalLM.from_pretrained(
        published_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(published_path)
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(device)
    positions = torch.arange(len(ids), device=device)
    mine = llm_7b.model_runner.model

    with torch.inference_mode():
        ref = hf(ids[None]).logits[0]
        hidden = mine(ids, positions)
        got = mine.compute_logits(hidden)

    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)
    assert int(got[-1].argmax()) == int(ref[-1].argmax())

    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    ours = llm_7b.generate([prompt], sampling_params=params)
    with torch.inference_mode():
        hf_ids = hf.generate(ids[None], max_new_tokens=8, do_sample=False)[0, ids.shape[0]:]
    assert ours[0].new_token_ids == hf_ids.tolist()
