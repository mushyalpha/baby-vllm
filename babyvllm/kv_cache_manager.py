from collections import deque
from babyvllm.sequence import Sequence

_DUMMY_BLOCK = 0

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0, "num_blocks must be > 0"
        assert block_size > 0, "block_size must be > 0"
        
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_block_ids = deque(range(num_blocks))

        self.used_block_ids = set()
        self.request_blocks: dict[int, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def reset(self):
        self.free_block_ids = deque(range(self.num_blocks))
        self.used_block_ids.clear()
        self.request_blocks.clear()

    def _check_invariants(self):
        assert self.num_free_blocks + len(self.used_block_ids) == self.num_blocks

    def get_block_table(self, seq: Sequence) -> list[int]:
        return self.request_blocks.get(seq.seq_id, [])

    def get_slot_mapping(self, seq: Sequence, num_new_tokens: int) -> list[int]:
        table = self.request_blocks.get(seq.seq_id, [])
        start = seq.num_computed_tokens
        out = []
        for pos in range(start, start + num_new_tokens):
            b, o = divmod(pos, self.block_size)
            assert b < len(table), f"seq {seq.seq_id} pos {pos} has no block"
            out.append(table[b] * self.block_size + o)
        return out

    def num_blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        target_num_computed_tokens = seq.num_computed_tokens + num_new_tokens
        target_num_logical_blocks = (target_num_computed_tokens + self.block_size - 1) // self.block_size
        return max(0, target_num_logical_blocks - len(self.request_blocks.get(seq.seq_id, [])))

    def can_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        return self.num_free_blocks >= self.num_blocks_needed(seq, num_new_tokens)

    def allocate_slots(self, seq: Sequence, num_new_tokens: int) -> None:
        needed = self.num_blocks_needed(seq, num_new_tokens)
        
        assert self.can_allocate(seq, num_new_tokens), f"OOM: cannot allocate for seq {seq.seq_id}. Free: {self.num_free_blocks}, needed: {needed}"

        if seq.seq_id not in self.request_blocks:
            self.request_blocks[seq.seq_id] = []

        for _ in range(needed):
            block_id = self.free_block_ids.popleft()
            self.used_block_ids.add(block_id)
            self.request_blocks[seq.seq_id].append(block_id)

    def free(self, seq: Sequence): 
        assert seq.seq_id in self.request_blocks, f"free() called on seq {seq.seq_id} with no allocated blocks. Double-free or free-without-allocate."

        for block_id in self.request_blocks[seq.seq_id]:
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)

        del self.request_blocks[seq.seq_id]

    def free_if_allocated(self, seq: Sequence):
        if seq.seq_id in self.request_blocks:
            self.free(seq)