"""CUDA graph decode: greedy argmax inside the graph and incremental inputs.

Qwen2.5-0.5B is the kernel-correctness model. Nothing here is a published number.
"""
import gc

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Write one sentence about paged attention and why block tables matter.",
    "1 2 3 4 5 6 7 8 9 10",
    "A " * 300,
]


def generate(use_graphs: bool, max_tokens: int = 40):
    from babyvllm.llm import LLM
    from babyvllm.sequence import SamplingParams

    llm = LLM(MODEL, use_cuda_graphs=use_graphs)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    outs = llm.generate(PROMPTS, params)
    ids = [o.new_token_ids for o in outs]
    graphs = len(llm.model_runner._graphs)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return ids, graphs


@pytest.mark.parametrize("mode", ["fp32", "bf16", "triton"])
def test_graph_greedy_tokens_match_eager_decode(mode, monkeypatch):
    eager, eager_graphs = generate(False)
    monkeypatch.setenv("BABYVLLM_GREEDY_ARGMAX", mode)
    graphed, n_graphs = generate(True)
    assert eager_graphs == 0
    assert n_graphs > 0, "decode never took the graph path"
    assert graphed == eager


@pytest.mark.parametrize("mode", ["fp32", "bf16", "triton"])
def test_graph_tokens_buffer_is_argmax_of_logits(mode, monkeypatch):
    from babyvllm.llm import LLM
    from babyvllm.sequence import SamplingParams

    monkeypatch.setenv("BABYVLLM_GREEDY_ARGMAX", mode)
    llm = LLM(MODEL, use_cuda_graphs=True)
    assert llm.model_runner.greedy_argmax == mode
    runner = llm.model_runner
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    for p in PROMPTS[:3]:
        llm.engine.add_request(llm.tokenizer(p).input_ids, params)
    seen = False
    while llm.engine.has_unfinished_requests():
        llm.engine.step()
        for buf in runner._static.values():
            n = buf.logits.shape[0]
            if bool(buf.logits.abs().sum() > 0):
                torch.testing.assert_close(buf.tokens[:n], buf.logits[:n].argmax(-1), atol=0, rtol=0)
                seen = True
    assert seen


@pytest.mark.parametrize("mode", ["fp32", "triton"])
def test_mixed_greedy_and_sampled_batch_keeps_greedy_rows_exact(mode, monkeypatch):
    from babyvllm.llm import LLM
    from babyvllm.sequence import SamplingParams

    monkeypatch.setenv("BABYVLLM_GREEDY_ARGMAX", mode)
    greedy = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    hot = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=24, ignore_eos=True)

    llm = LLM(MODEL, use_cuda_graphs=True)
    ref = [o.new_token_ids for o in llm.generate(PROMPTS[:3], greedy)]

    ids = [llm.tokenizer(p).input_ids for p in PROMPTS[:3]]
    seq_ids = [
        llm.engine.add_request(ids[0], greedy),
        llm.engine.add_request(ids[1], hot),
        llm.engine.add_request(ids[2], greedy),
    ]
    outs = llm.engine.run()
    assert outs[seq_ids[0]].new_token_ids == ref[0]
    assert outs[seq_ids[2]].new_token_ids == ref[2]
    assert len(outs[seq_ids[1]].new_token_ids) == 24
