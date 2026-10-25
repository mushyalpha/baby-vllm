import pytest

from babyvllm.llm import LLM
from babyvllm.sequence import SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B"

SYSTEM = (
    "You are a careful assistant for a small engineering team. Answer in one "
    "short paragraph, name the assumption you are least sure of, and never "
    "invent file names, numbers, or citations that were not given to you. "
    "If a question is ambiguous, say which reading you chose and why."
)
QUESTIONS = [
    " Question: why does a paged KV cache reduce fragmentation?",
    " Question: what is a CUDA graph?",
    " Question: when is greedy decoding a bad idea?",
]


@pytest.fixture(scope="module")
def llm():
    return LLM(MODEL, device="cpu")


def generate(engine, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    ids = [engine.add_request(p, params) for p in prompts]
    outs = engine.run()
    return [outs[i].new_token_ids for i in ids]


def test_greedy_outputs_match_with_and_without_caching(llm):
    engine = llm.engine
    kv = engine.kv_cache_manager
    bs = kv.block_size

    system = llm.tokenizer(SYSTEM).input_ids
    assert len(system) > 2 * bs
    prompts = [system + llm.tokenizer(q).input_ids for q in QUESTIONS]
    # Two full system blocks plus one token. Once those blocks are cached,
    # this prompt has a single token left and runs through decode attention.
    one_left = system[: 2 * bs + 1]
    cases = prompts + [one_left]

    kv.enable_prefix_caching = False
    kv.reset()
    ref = generate(engine, cases, max_tokens=8)

    kv.enable_prefix_caching = True
    kv.reset()
    try:
        same_step = generate(engine, cases, max_tokens=8)
        hits_same_step = kv.num_hit_tokens
        later = generate(engine, cases, max_tokens=8)
        alone = generate(engine, [one_left], max_tokens=8)
    finally:
        kv.enable_prefix_caching = False
        kv.reset()

    assert hits_same_step > 0
    assert same_step == ref
    assert later == ref
    assert alone == [ref[-1]]
