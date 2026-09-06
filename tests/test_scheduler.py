import pytest
from babyvllm.scheduler import Scheduler
from babyvllm.sequence import Sequence
from babyvllm.kv_cache_manager import KVCacheManager

def test_scheduler_preempt_no_double_schedule():
    kv_cache = KVCacheManager(num_blocks=2, block_size=1)
    scheduler = Scheduler(kv_cache_manager=kv_cache, max_num_batched_tokens=32, max_num_seqs=2)
    
    seq1 = Sequence([1])
    seq2 = Sequence([2])
    
    scheduler.add_seq(seq1)
    scheduler.add_seq(seq2)
    
    out1 = scheduler.schedule()
    assert len(out1.scheduled_sequences) == 2
    assert len(scheduler.running) == 2
    
    scheduler.update_from_output(out1, {seq1.seq_id: 3, seq2.seq_id: 4})
    
    out2 = scheduler.schedule()
    
    try:
        scheduler.update_from_output(out2, {seq1.seq_id: 5})
    except ValueError as e:
        pytest.fail(f"Double scheduling bug occurred: {e}")
        
    assert len(out2.scheduled_sequences) == 1

import random
from babyvllm.engine import LLMEngine
from babyvllm.sequence import SamplingParams, SequenceFinishReason

class FakeModelRunner:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.num_blocks = 100
        self.eos_id = 999
        
    def determine_num_blocks(self):
        return self.num_blocks
        
    def execute_model(self, scheduler_output):
        sampled_tokens = {}
        for seq in scheduler_output.scheduled_sequences:
            token = (seq.seq_id + len(seq)) % 1000
            if len(seq) >= 15:
                token = self.eos_id
            sampled_tokens[seq.seq_id] = token
        return sampled_tokens

def run_stress_test(num_blocks):
    runner = FakeModelRunner(block_size=4)
    runner.num_blocks = num_blocks
    engine = LLMEngine(model_runner=runner, max_num_batched_tokens=32, max_num_seqs=8)
    
    original_schedule = engine.scheduler.schedule
    def hooked_schedule():
        out = original_schedule()
        seq_ids = [s.seq_id for s in out.scheduled_sequences]
        assert len(seq_ids) == len(set(seq_ids)), "Duplicate sequence in scheduled_sequences!"
        
        for seq in out.scheduled_sequences:
            needed = out.num_scheduled_tokens[seq.seq_id]
            assert len(out.slot_mappings[seq.seq_id]) == needed, "Slot mapping length mismatch!"
        return out
    engine.scheduler.schedule = hooked_schedule
    
    random.seed(42)
    for i in range(20):
        prompt_len = random.randint(1, 10)
        prompt = [random.randint(0, 100) for _ in range(prompt_len)]
        max_tokens = random.randint(5, 20)
        engine.add_request(prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True))
        
    outputs = {}
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for out in step_outputs:
            if out.seq_id not in outputs:
                outputs[out.seq_id] = []
            outputs[out.seq_id].extend(out.new_token_ids)
            
        assert engine.kv_cache_manager.num_free_blocks + len(engine.kv_cache_manager.used_block_ids) == runner.num_blocks
        
        for seq in engine.scheduler.running:
            assert seq.num_computed_tokens <= len(seq)
            
    return outputs

def test_invariant_stress_test():
    out_large = run_stress_test(num_blocks=1000)
    out_small = run_stress_test(num_blocks=10)
    
    assert len(out_large) == 20
    assert len(out_small) == 20
    
    large_vals = [out_large[k] for k in sorted(out_large.keys())]
    small_vals = [out_small[k] for k in sorted(out_small.keys())]
    assert large_vals == small_vals, "Outputs mismatch under preemption!"

def test_eos_stop():
    runner = FakeModelRunner(block_size=4)
    engine = LLMEngine(model_runner=runner, max_num_batched_tokens=32, max_num_seqs=8)
    
    prompt = [1, 2, 3]
    engine.add_request(prompt, SamplingParams(max_tokens=50, stop_tokens={runner.eos_id}, ignore_eos=False))
    
    outputs = engine.run()
    seq_out = list(outputs.values())[0]
    
    assert seq_out.finish_reason == SequenceFinishReason.STOP
    assert runner.eos_id not in seq_out.new_token_ids
