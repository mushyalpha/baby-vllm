import itertools
import random

import pytest

from babyvllm.engine import LLMEngine
from babyvllm.kv_cache_manager import KVCacheManager
from babyvllm.sequence import Sequence, SamplingParams


def admit(kv: KVCacheManager, seq: Sequence) -> list[int]:
    """Admit ``seq`` the way the scheduler does, then mark its prompt computed."""
    computed = kv.get_computed_blocks(seq)
    num_cached = len(computed) * kv.block_size
    kv.allocate_slots(seq, len(seq) - num_cached, computed)
    seq.advance_computed(len(seq))
    return computed


def test_same_prefix_shares_full_blocks():
    kv = KVCacheManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    a = Sequence(prefix + [9, 10])
    b = Sequence(prefix + [11, 12])

    assert admit(kv, a) == []
    a_table = kv.get_block_table(a)
    assert admit(kv, b) == a_table[:2]

    b_table = kv.get_block_table(b)
    assert b_table[:2] == a_table[:2]
    assert b_table[2] != a_table[2]
    assert [kv.ref_counts[x] for x in a_table] == [2, 2, 1]
    assert kv.num_hit_tokens == 8
    assert kv.num_queried_tokens == 20
    kv._check_invariants()

    kv.free(a)
    assert [kv.ref_counts[x] for x in b_table] == [1, 1, 1]
    kv._check_invariants()


def test_partial_block_is_not_shared():
    kv = KVCacheManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    a = Sequence([1, 2, 3, 4, 5, 6])
    admit(kv, a)
    assert kv.block_keys[kv.get_block_table(a)[1]] is None

    b = Sequence([1, 2, 3, 4, 5, 6, 7])
    assert admit(kv, b) == kv.get_block_table(a)[:1]
    kv._check_invariants()


def test_identical_prompt_recomputes_last_block():
    kv = KVCacheManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    a = Sequence(prompt)
    b = Sequence(prompt)
    admit(kv, a)

    computed = kv.get_computed_blocks(b)
    assert len(computed) == 1
    kv.allocate_slots(b, len(b) - 4, computed)
    b.advance_computed(4)
    # The recomputed tokens go to a private block, never into a's block 1.
    a_table, b_table = kv.get_block_table(a), kv.get_block_table(b)
    assert b_table[0] == a_table[0]
    assert b_table[1] != a_table[1]
    assert kv.get_slot_mapping(b, 4)[0] == b_table[1] * 4
    # The duplicate content stays unregistered; a's block keeps the key.
    assert kv.block_keys[b_table[1]] is None
    kv._check_invariants()


def test_freed_blocks_stay_cached_until_evicted():
    kv = KVCacheManager(num_blocks=4, block_size=4, enable_prefix_caching=True)
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    a = Sequence(prefix + [9])
    admit(kv, a)
    a_table = list(kv.get_block_table(a))
    kv.free(a)
    assert kv.num_free_blocks == 4
    kv._check_invariants()

    c = Sequence(prefix + [50])
    assert admit(kv, c) == a_table[:2]
    kv.free(c)

    d = Sequence(list(range(100, 116)))
    admit(kv, d)
    kv.free(d)
    kv._check_invariants()

    e = Sequence(prefix + [60])
    assert kv.get_computed_blocks(e) == []


def test_can_allocate_counts_revived_blocks():
    kv = KVCacheManager(num_blocks=3, block_size=4, enable_prefix_caching=True)
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    a = Sequence(prefix + [9])
    admit(kv, a)
    kv.free(a)

    y = Sequence([77])
    admit(kv, y)
    assert kv.num_free_blocks == 2

    b = Sequence(prefix + [42])
    computed = kv.get_computed_blocks(b)
    assert len(computed) == 2
    # One new block plus two hits that are on the free list is three blocks.
    assert not kv.can_allocate(b, 1, computed)

    kv.free(y)
    assert kv.can_allocate(b, 1, computed)
    kv.allocate_slots(b, 1, computed)
    assert kv.num_free_blocks == 0
    kv._check_invariants()


def test_disabled_never_shares():
    kv = KVCacheManager(num_blocks=16, block_size=4)
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    a = Sequence(prefix + [9])
    b = Sequence(prefix + [10])
    admit(kv, a)
    assert admit(kv, b) == []
    assert not set(kv.get_block_table(a)) & set(kv.get_block_table(b))
    assert kv.cached_blocks == {}


class KVCheckingRunner:
    """Writes token ids into a fake KV pool through the slot mapping, then
    reads each sequence's whole context back through its block table.

    The sampled token depends on the context read back, so a wrong block
    table or an overwritten shared block changes the outputs.
    """

    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.pool = [None] * (num_blocks * block_size)

    def allocate_kv_cache(self):
        return self.num_blocks

    def execute_model(self, scheduler_output):
        bs = self.block_size
        for seq in scheduler_output.scheduled_sequences:
            start = seq.num_computed_tokens
            n = scheduler_output.num_scheduled_tokens[seq.seq_id]
            for slot, tok in zip(scheduler_output.slot_mappings[seq.seq_id], seq.token_ids[start:start + n]):
                self.pool[slot] = tok

        out = {}
        for seq in scheduler_output.scheduled_sequences:
            seq_len = seq.num_computed_tokens + scheduler_output.num_scheduled_tokens[seq.seq_id]
            table = scheduler_output.block_tables[seq.seq_id]
            ctx = [self.pool[table[p // bs] * bs + p % bs] for p in range(seq_len)]
            assert ctx == seq.token_ids[:seq_len], f"seq {seq.seq_id} read back the wrong context"
            out[seq.seq_id] = (sum(ctx) * 31 + seq_len) % 1000
        return out


def shared_prefix_requests(seed: int, n: int):
    rng = random.Random(seed)
    system_prompts = [[rng.randint(0, 100) for _ in range(rng.randint(6, 13))] for _ in range(3)]
    requests = []
    for _ in range(n):
        prompt = rng.choice(system_prompts) + [rng.randint(0, 100) for _ in range(rng.randint(1, 5))]
        requests.append((prompt, rng.randint(3, 12)))
    return requests


def run_engine(requests, num_blocks: int, enable_prefix_caching: bool, stagger: bool):
    Sequence._counter = itertools.count()
    runner = KVCheckingRunner(block_size=4, num_blocks=num_blocks)
    engine = LLMEngine(
        model_runner=runner,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        enable_prefix_caching=enable_prefix_caching,
    )
    pending = list(requests)
    first = pending[: len(pending) // 2] if stagger else pending
    rest = pending[len(first):]
    for prompt, max_tokens in first:
        engine.add_request(prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True))

    outputs: dict[int, list[int]] = {}
    steps = 0
    while engine.has_unfinished_requests() or rest:
        if rest and steps == 3:
            for prompt, max_tokens in rest:
                engine.add_request(prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True))
            rest = []
        for out in engine.step():
            outputs.setdefault(out.seq_id, []).extend(out.new_token_ids)
        engine.kv_cache_manager._check_invariants()
        steps += 1
        assert steps < 10_000
    return outputs, engine.kv_cache_manager


@pytest.mark.parametrize("num_blocks", [200, 12])
@pytest.mark.parametrize("stagger", [False, True])
def test_engine_outputs_match_with_and_without_caching(num_blocks, stagger):
    requests = shared_prefix_requests(seed=0, n=24)
    off, _ = run_engine(requests, num_blocks, enable_prefix_caching=False, stagger=stagger)
    on, kv = run_engine(requests, num_blocks, enable_prefix_caching=True, stagger=stagger)

    assert len(on) == len(requests)
    assert on == off
    assert kv.num_hit_tokens > 0
    assert kv.num_free_blocks == num_blocks
