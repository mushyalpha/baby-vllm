from collections import OrderedDict
from babyvllm.sequence import Sequence

_DUMMY_BLOCK = 0

# (hash of the previous block, this block's token ids). The token tuple makes
# a dict lookup exact for the block itself; only the parent link is a hash.
BlockKey = tuple[int | None, tuple[int, ...]]


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = False):
        assert num_blocks > 0, "num_blocks must be > 0"
        assert block_size > 0, "block_size must be > 0"

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.reset()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def prefix_cache_hit_rate(self) -> float:
        if self.num_queried_tokens == 0:
            return 0.0
        return self.num_hit_tokens / self.num_queried_tokens

    def reset(self):
        # Front is least recently freed, so it is evicted first. A block with
        # ref count 0 stays here even while it is still in cached_blocks.
        self.free_block_ids: OrderedDict[int, None] = OrderedDict.fromkeys(range(self.num_blocks))
        self.used_block_ids: set[int] = set()
        self.ref_counts = [0] * self.num_blocks
        self.block_keys: list[BlockKey | None] = [None] * self.num_blocks
        self.cached_blocks: dict[BlockKey, int] = {}
        self.request_blocks: dict[int, list[int]] = {}
        self.num_registered_blocks: dict[int, int] = {}
        self.num_queried_tokens = 0
        self.num_hit_tokens = 0

    def _check_invariants(self):
        assert self.num_free_blocks + len(self.used_block_ids) == self.num_blocks
        for b in range(self.num_blocks):
            assert (self.ref_counts[b] == 0) == (b in self.free_block_ids), f"block {b} ref/free mismatch"
            key = self.block_keys[b]
            if key is not None:
                assert self.cached_blocks[key] == b, f"block {b} key not mapped back"
        for key, b in self.cached_blocks.items():
            assert self.block_keys[b] == key, f"cached key maps to block {b} with a different key"
        refs = [0] * self.num_blocks
        for table in self.request_blocks.values():
            for b in table:
                refs[b] += 1
        assert refs == self.ref_counts, "ref counts disagree with block tables"

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

    def _block_key(self, seq: Sequence, i: int) -> BlockKey:
        parent = seq.block_hashes[i - 1] if i > 0 else None
        start = i * self.block_size
        return (parent, tuple(seq.token_ids[start:start + self.block_size]))

    def _update_block_hashes(self, seq: Sequence, num_tokens: int) -> None:
        num_full = min(num_tokens, len(seq)) // self.block_size
        while len(seq.block_hashes) < num_full:
            seq.block_hashes.append(hash(self._block_key(seq, len(seq.block_hashes))))

    def get_computed_blocks(self, seq: Sequence) -> list[int]:
        """Longest run of cached full blocks at the start of a new sequence.

        Read-only. At most ``(len(seq) - 1) // block_size`` blocks, so at least
        one prompt token is computed and produces logits, and that token never
        lands in a shared block.
        """
        if not self.enable_prefix_caching or seq.seq_id in self.request_blocks:
            return []
        max_blocks = (len(seq) - 1) // self.block_size
        self._update_block_hashes(seq, max_blocks * self.block_size)
        out = []
        for i in range(max_blocks):
            b = self.cached_blocks.get(self._block_key(seq, i))
            if b is None:
                break
            out.append(b)
        return out

    def num_blocks_needed(self, seq: Sequence, num_new_tokens: int, computed_blocks: list[int] = ()) -> int:
        num_computed = seq.num_computed_tokens + len(computed_blocks) * self.block_size
        target_num_logical_blocks = (num_computed + num_new_tokens + self.block_size - 1) // self.block_size
        num_held = len(self.request_blocks.get(seq.seq_id, [])) + len(computed_blocks)
        return max(0, target_num_logical_blocks - num_held)

    def can_allocate(self, seq: Sequence, num_new_tokens: int, computed_blocks: list[int] = ()) -> bool:
        # A hit on a block with ref count 0 takes it off the free list too.
        num_revived = sum(1 for b in computed_blocks if self.ref_counts[b] == 0)
        return self.num_free_blocks >= self.num_blocks_needed(seq, num_new_tokens, computed_blocks) + num_revived

    def _pop_free_block(self) -> int:
        block_id, _ = self.free_block_ids.popitem(last=False)
        key = self.block_keys[block_id]
        if key is not None:
            del self.cached_blocks[key]
            self.block_keys[block_id] = None
        return block_id

    def _acquire(self, block_id: int) -> None:
        if self.ref_counts[block_id] == 0:
            del self.free_block_ids[block_id]
            self.used_block_ids.add(block_id)
        self.ref_counts[block_id] += 1

    def allocate_slots(self, seq: Sequence, num_new_tokens: int, computed_blocks: list[int] = ()) -> None:
        """Give ``seq`` blocks for ``num_new_tokens`` more tokens.

        ``computed_blocks`` comes from ``get_computed_blocks`` on a sequence
        with no blocks yet. The caller then advances ``num_computed_tokens``
        past them before building the slot mapping.
        """
        needed = self.num_blocks_needed(seq, num_new_tokens, computed_blocks)

        assert self.can_allocate(seq, num_new_tokens, computed_blocks), f"OOM: cannot allocate for seq {seq.seq_id}. Free: {self.num_free_blocks}, needed: {needed}"

        table = self.request_blocks.setdefault(seq.seq_id, [])
        if computed_blocks:
            assert len(table) == 0 and seq.num_computed_tokens == 0, f"seq {seq.seq_id} already has blocks"
            for block_id in computed_blocks:
                self._acquire(block_id)
                table.append(block_id)

        for _ in range(needed):
            block_id = self._pop_free_block()
            self.ref_counts[block_id] = 1
            self.used_block_ids.add(block_id)
            table.append(block_id)

        if self.enable_prefix_caching:
            num_cached = len(computed_blocks) * self.block_size
            if seq.num_computed_tokens == 0:
                self.num_queried_tokens += num_cached + num_new_tokens
                self.num_hit_tokens += num_cached
            # Registered before the forward pass writes them. A sequence that
            # hits these in the same step reads them after the same layer's
            # store, which runs for every token before attention.
            self.cache_full_blocks(seq, seq.num_computed_tokens + num_cached + num_new_tokens)

    def cache_full_blocks(self, seq: Sequence, num_tokens: int) -> None:
        table = self.request_blocks[seq.seq_id]
        num_full = num_tokens // self.block_size
        self._update_block_hashes(seq, num_full * self.block_size)
        start = self.num_registered_blocks.get(seq.seq_id, 0)
        for i in range(start, num_full):
            block_id = table[i]
            if self.block_keys[block_id] is not None:
                continue
            key = self._block_key(seq, i)
            # Another block already holds this content; this one stays private.
            if key not in self.cached_blocks:
                self.cached_blocks[key] = block_id
                self.block_keys[block_id] = key
        self.num_registered_blocks[seq.seq_id] = max(start, num_full)

    def free(self, seq: Sequence):
        assert seq.seq_id in self.request_blocks, f"free() called on seq {seq.seq_id} with no allocated blocks. Double-free or free-without-allocate."

        # Tail first, so a sequence's own suffix is evicted before a prefix
        # other requests may share.
        for block_id in reversed(self.request_blocks[seq.seq_id]):
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids[block_id] = None

        del self.request_blocks[seq.seq_id]
        self.num_registered_blocks.pop(seq.seq_id, None)

    def free_if_allocated(self, seq: Sequence):
        if seq.seq_id in self.request_blocks:
            self.free(seq)
