import random

import numpy as np
import pytest
import torch

from babyvllm.kv_cache_manager import KVCacheManager
from babyvllm.scheduler import Scheduler
from babyvllm.sequence import SamplingParams, Sequence
from babyvllm.worker.decode_inputs import DecodeInputs
from babyvllm.worker.model_runner import BATCH_BUCKETS, ceil_to_bucket

BS = 16
MB = 32
DUMMY = 9999


def reference(seqs, out, bucket):
    """The old prepare_static_inputs, rebuilt from scratch every step."""
    ids = np.zeros(bucket, dtype=np.int64)
    pos = np.zeros(bucket, dtype=np.int64)
    seq_lens = np.zeros(bucket, dtype=np.int32)
    ctx = np.zeros(bucket, dtype=np.int32)
    slots = np.full(bucket, DUMMY * BS, dtype=np.int64)
    tables = np.full((bucket, MB), DUMMY, dtype=np.int32)
    for i, s in enumerate(seqs):
        start = s.num_computed_tokens
        ids[i] = s.token_ids[start]
        pos[i] = start
        seq_lens[i] = start + 1
        ctx[i] = start
        slots[i] = out.slot_mappings[s.seq_id][0]
        t = out.block_tables[s.seq_id]
        tables[i, : len(t)] = t
    return ids, pos, seq_lens, ctx, slots, tables


def check(host, seqs, out):
    bucket = ceil_to_bucket(len(seqs), BATCH_BUCKETS)
    host.update(seqs, out.block_tables, bucket)
    ids, pos, seq_lens, ctx, slots, tables = reference(seqs, out, bucket)
    assert np.array_equal(host.input_ids[:bucket], ids)
    assert np.array_equal(host.positions[:bucket], pos)
    assert np.array_equal(host.seq_lens[:bucket], seq_lens)
    assert np.array_equal(host.context_lens[:bucket], ctx)
    assert np.array_equal(host.slot_mapping[:bucket], slots)
    assert np.array_equal(host.block_tables[:bucket], tables)


def run(seed, num_blocks, prefix_caching, n_requests, max_num_seqs):
    rng = random.Random(seed)
    kv = KVCacheManager(num_blocks=num_blocks, block_size=BS, enable_prefix_caching=prefix_caching)
    sch = Scheduler(kv, max_num_batched_tokens=512, max_num_seqs=max_num_seqs)
    host = DecodeInputs(64, MB, BS, DUMMY, pin=False)
    shared = [rng.randrange(10, 900) for _ in range(2 * BS)]
    for _ in range(n_requests):
        if rng.random() < 0.6:
            prompt = shared + [rng.randrange(10, 900) for _ in range(rng.randrange(1, 40))]
        else:
            prompt = [rng.randrange(10, 900) for _ in range(rng.randrange(5, 120))]
        sch.add_seq(Sequence(prompt, SamplingParams(temperature=0.0, max_tokens=rng.randrange(1, 70), ignore_eos=True)))

    decode_steps = preempted = 0
    for _ in range(3000):
        if not sch.has_unfinished():
            break
        out = sch.schedule()
        preempted += len(out.preempted_seqs)
        seqs = out.scheduled_sequences
        if seqs and all(out.num_scheduled_tokens[s.seq_id] == 1 for s in seqs):
            check(host, seqs, out)
            decode_steps += 1
        tokens = {s.seq_id: 10 + (s.seq_id * 7 + len(s)) % 800 for s in seqs}
        sch.update_from_output(out, tokens)
        sch.pop_finished()
    assert not sch.has_unfinished()
    return decode_steps, preempted


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_incremental_matches_full_rebuild_with_plenty_of_memory(prefix_caching):
    steps, _ = run(seed=1, num_blocks=2000, prefix_caching=prefix_caching, n_requests=40, max_num_seqs=64)
    assert steps > 50


@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_matches_full_rebuild_under_preemption(seed, prefix_caching):
    steps, preempted = run(seed=seed, num_blocks=26, prefix_caching=prefix_caching, n_requests=30, max_num_seqs=12)
    assert steps > 20
    assert preempted > 0, "this test must actually preempt"


def test_batch_shrinking_clears_dropped_rows():
    host = DecodeInputs(8, MB, BS, DUMMY, pin=False)
    kv = KVCacheManager(num_blocks=200, block_size=BS)
    sch = Scheduler(kv, max_num_batched_tokens=512, max_num_seqs=8)
    for i, n in enumerate([20, 20, 20, 20]):
        sch.add_seq(Sequence([10 + i] * n, SamplingParams(temperature=0.0, max_tokens=3 + 5 * i, ignore_eos=True)))
    out = sch.schedule()
    sch.update_from_output(out, {s.seq_id: 11 for s in out.scheduled_sequences})
    sizes = []
    while sch.has_unfinished():
        out = sch.schedule()
        seqs = out.scheduled_sequences
        check(host, seqs, out)
        sizes.append(len(seqs))
        sch.update_from_output(out, {s.seq_id: 11 for s in seqs})
        sch.pop_finished()
    assert sizes[0] == 4 and sizes[-1] == 1


def test_slot_mapping_fast_path_matches_the_loop():
    kv = KVCacheManager(num_blocks=50, block_size=BS)
    seq = Sequence(list(range(10, 60)), SamplingParams())
    kv.allocate_slots(seq, 50)
    seq.advance_computed(50)
    seq.append_token(5)
    kv.allocate_slots(seq, 1)
    table = kv.get_block_table(seq)
    assert kv.get_slot_mapping(seq, 1) == [table[50 // BS] * BS + 50 % BS]


def test_preemption_changes_the_row_key():
    seq = Sequence([1, 2, 3], SamplingParams())
    before = (seq.seq_id, seq.num_preemptions)
    seq.reset_for_recompute()
    assert (seq.seq_id, seq.num_preemptions) != before


def test_staging_is_plain_tensors_off_cuda():
    host = DecodeInputs(4, MB, BS, DUMMY, pin=False)
    assert isinstance(host.t_block_tables, torch.Tensor)
    assert host.block_tables.shape == (4, MB)


def test_recomputed_sequence_with_same_table_length_is_rewritten():
    host = DecodeInputs(4, MB, BS, DUMMY, pin=False)
    seq = Sequence(list(range(10, 30)), SamplingParams())
    seq.advance_computed(20)
    seq.append_token(7)
    host.update([seq], {seq.seq_id: [1, 2]}, 1)
    assert list(host.block_tables[0, :3]) == [1, 2, DUMMY]
    # Preempted and recomputed into different blocks of the same count, and
    # back in the same row with the same sequence id.
    seq.reset_for_recompute()
    seq.advance_computed(20)
    host.update([seq], {seq.seq_id: [5, 6]}, 1)
    assert list(host.block_tables[0, :3]) == [5, 6, DUMMY]
    assert host.slot_mapping[0] == 6 * BS + 4
