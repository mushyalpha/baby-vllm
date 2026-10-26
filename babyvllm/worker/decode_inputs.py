"""Host-side inputs for a CUDA-graph decode step, updated in place.

The graph reads static GPU buffers. Each step the host has to put the next
token, position, sequence length, slot and block table for every sequence into
them. The old path rebuilt Python lists and made one ``torch.tensor`` per
sequence for the block tables. In steady decode almost none of that changes:
every position moves by one, and a sequence's block table grows by one block
every ``block_size`` tokens.

This keeps one set of staging arrays for the largest batch bucket and edits
them. A row is rewritten from the scheduler's block table only when the
sequence in that row changed, was preempted and recomputed, or got a new
block. Every other step touches a handful of vectorised numpy slices.

The arrays are numpy views of pinned CPU tensors (plain tensors off CUDA), so
the same bytes feed the non-blocking host-to-device copies.
"""

from __future__ import annotations

import numpy as np
import torch


class DecodeInputs:
    def __init__(
        self,
        max_batch: int,
        max_blocks: int,
        block_size: int,
        dummy_block: int,
        pin: bool,
    ):
        self.max_batch = max_batch
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.dummy_block = dummy_block
        self.dummy_slot = dummy_block * block_size

        def host(shape, dtype, fill):
            t = torch.full(shape, fill, dtype=dtype, pin_memory=pin)
            return t, t.numpy()

        self.t_input_ids, self.input_ids = host((max_batch,), torch.int64, 0)
        self.t_positions, self.positions = host((max_batch,), torch.int64, 0)
        self.t_seq_lens, self.seq_lens = host((max_batch,), torch.int32, 0)
        self.t_context_lens, self.context_lens = host((max_batch,), torch.int32, 0)
        self.t_slot_mapping, self.slot_mapping = host((max_batch,), torch.int64, self.dummy_slot)
        self.t_block_tables, self.block_tables = host((max_batch, max_blocks), torch.int32, dummy_block)
        # Sampled tokens come back through here after the graph runs.
        self.t_tokens, self.tokens = host((max_batch,), torch.int64, 0)

        # What row i currently holds: (seq_id, preemption epoch) and how many
        # blocks of its table are written. None means a dummy row.
        self._row_key: list[tuple[int, int] | None] = [None] * max_batch
        self._row_blocks = [0] * max_batch
        self._n = 0
        self._rows = np.arange(max_batch)

    def update(self, seqs, block_tables: dict[int, list[int]], bucket: int) -> int:
        """Bring the staging arrays to this step's decode batch.

        Every sequence in ``seqs`` has exactly one token to compute, at
        ``seq.num_computed_tokens``. Returns the number of live rows.
        """
        n = len(seqs)
        if n > bucket or bucket > self.max_batch:
            raise ValueError(f"batch {n} does not fit bucket {bucket} (max {self.max_batch})")
        bs = self.block_size

        pos = [s.num_computed_tokens for s in seqs]
        self.positions[:n] = pos
        self.input_ids[:n] = [s.token_ids[p] for s, p in zip(seqs, pos)]
        np.add(self.positions[:n], 1, out=self.seq_lens[:n], casting="unsafe")
        self.context_lens[:n] = self.positions[:n]

        # Rows that held a sequence last step and are padding now.
        for i in range(n, self._n):
            self._row_key[i] = None
            self._row_blocks[i] = 0
            self.block_tables[i].fill(self.dummy_block)

        row_key = self._row_key
        row_blocks = self._row_blocks
        bt = self.block_tables
        for i, s in enumerate(seqs):
            table = block_tables[s.seq_id]
            m = len(table)
            key = (s.seq_id, s.num_preemptions)
            if row_key[i] != key:
                bt[i, :m] = table
                bt[i, m:] = self.dummy_block
                row_key[i] = key
                row_blocks[i] = m
            elif row_blocks[i] != m:
                have = row_blocks[i]
                bt[i, have:m] = table[have:]
                row_blocks[i] = m
        self._n = n

        # The scheduler's slot for a one-token step, from the table row.
        rows = self._rows[:n]
        pos_np = self.positions[:n]
        self.slot_mapping[:n] = bt[rows, pos_np // bs].astype(np.int64) * bs + pos_np % bs

        if n < bucket:
            self.input_ids[n:bucket] = 0
            self.positions[n:bucket] = 0
            self.seq_lens[n:bucket] = 0
            self.context_lens[n:bucket] = 0
            self.slot_mapping[n:bucket] = self.dummy_slot
        return n
