from collections import deque
from dataclasses import dataclass
from babyvllm.sequence import Sequence, SequenceState, SequenceFinishReason
from babyvllm.kv_cache_manager import KVCacheManager

@dataclass
class SchedulerOutput:
    scheduled_sequences: list[Sequence]
    num_scheduled_tokens: dict[int, int]
    block_tables: dict[int, list[int]]
    slot_mappings: dict[int, list[int]]
    preempted_seqs: list[Sequence]

class Scheduler:
    def __init__(self, kv_cache_manager: KVCacheManager, max_num_batched_tokens: int = 2048, max_num_seqs: int = 256):
        
        
        self.kv_cache_manager = kv_cache_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished_seqs: list[Sequence] = []

    def _finish(self, seq: Sequence, reason: SequenceFinishReason) -> None:
        seq.finish(reason)
        self.free_seq(seq)
        self.finished_seqs.append(seq)

    def pop_finished(self) -> list[Sequence]:
        out, self.finished_seqs = self.finished_seqs, []
        return out

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def add_seq(self, seq: Sequence):
        if len(seq) > self.max_num_batched_tokens:
            self._finish(seq, SequenceFinishReason.LENGTH)
            return
            
        if self.kv_cache_manager.num_blocks_needed(seq, len(seq)) > self.kv_cache_manager.num_blocks:
            self._finish(seq, SequenceFinishReason.LENGTH)
            return
            
        seq.status = SequenceState.WAITING
        self.waiting.append(seq)

    def admit_seq(self, seq: Sequence):
        popped = self.waiting.popleft()
        assert popped == seq
        seq.status = SequenceState.RUNNING
        self.running.append(seq)

    def abort_seq(self, seq: Sequence):
        self._finish(seq, SequenceFinishReason.ABORT)

    def free_seq(self, seq: Sequence):
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        
        self.kv_cache_manager.free_if_allocated(seq)

    def update_from_output(self, scheduler_output: SchedulerOutput, model_output: dict[int, int]):
        for seq in scheduler_output.scheduled_sequences:
            num_scheduled = scheduler_output.num_scheduled_tokens[seq.seq_id]
            seq.advance_computed(num_scheduled)
            if seq.seq_id in model_output:
                token_id = model_output[seq.seq_id]
                if token_id in seq.sampling_params.stop_tokens:
                    self._finish(seq, SequenceFinishReason.STOP)
                else:
                    seq.append_token(token_id)
                    if seq.generated_token_len >= seq.sampling_params.max_tokens:
                        self._finish(seq, SequenceFinishReason.LENGTH)

    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        scheduled_sequences: list[Sequence] = []
        num_scheduled_tokens: dict[int, int] = {}
        block_tables: dict[int, list[int]] = {}
        slot_mappings: dict[int, list[int]] = {}

        running_seqs = list(self.running)
        preempted_seqs = []
        for seq in running_seqs:
            if seq.status != SequenceState.RUNNING:
                continue
            needed = seq.num_tokens_to_compute
            
            if budget < needed or len(scheduled_sequences) >= self.max_num_seqs:
                break
                
            while not self.kv_cache_manager.can_allocate(seq, needed):
                if len(self.running) > 1 and self.running[-1] is not seq:
                    victim = self.running[-1]
                    victim.reset_for_recompute()
                    self.running.remove(victim)
                    preempted_seqs.append(victim)
                    self.kv_cache_manager.free_if_allocated(victim)
                else:
                    seq.reset_for_recompute()
                    self.running.remove(seq)
                    preempted_seqs.append(seq)
                    self.kv_cache_manager.free_if_allocated(seq)
                    break
            else:
                self.kv_cache_manager.allocate_slots(seq, needed)
                scheduled_sequences.append(seq)
                num_scheduled_tokens[seq.seq_id] = needed
                block_tables[seq.seq_id] = list(self.kv_cache_manager.get_block_table(seq))
                slot_mappings[seq.seq_id] = self.kv_cache_manager.get_slot_mapping(seq, needed)
                budget -= needed

        for seq in reversed(preempted_seqs):
            self.waiting.appendleft(seq)

        while self.waiting:
            seq = self.waiting[0]
            if len(scheduled_sequences) >= self.max_num_seqs:
                break
                
            needed = seq.num_tokens_to_compute
            
            if needed > self.max_num_batched_tokens:
                self._finish(seq, SequenceFinishReason.LENGTH)
                continue
            
            if self.kv_cache_manager.num_blocks_needed(seq, needed) > self.kv_cache_manager.num_blocks:
                self._finish(seq, SequenceFinishReason.LENGTH)
                continue
            
            if budget >= needed and self.kv_cache_manager.can_allocate(seq, needed):
                self.kv_cache_manager.allocate_slots(seq, needed)
                self.admit_seq(seq)
                scheduled_sequences.append(seq)
                num_scheduled_tokens[seq.seq_id] = needed
                block_tables[seq.seq_id] = list(self.kv_cache_manager.get_block_table(seq))
                slot_mappings[seq.seq_id] = self.kv_cache_manager.get_slot_mapping(seq, needed)
                budget -= needed
            else:
                break

        return SchedulerOutput(
            scheduled_sequences=scheduled_sequences,
            num_scheduled_tokens=num_scheduled_tokens,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            preempted_seqs=preempted_seqs,
        )

