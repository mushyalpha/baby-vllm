import re

with open("babyvllm/worker/model_runner.py", "r") as f:
    content = f.read()

# Add cpu buffers to StaticBuffers
new_class = """class StaticBuffers:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    logits: torch.Tensor
    # Pinned CPU staging buffers
    cpu_input_ids: torch.Tensor
    cpu_position_ids: torch.Tensor
    cpu_seq_lens: torch.Tensor
    cpu_context_lens: torch.Tensor
    cpu_slot_mapping: torch.Tensor
    cpu_block_tables: torch.Tensor
"""
content = re.sub(r"class StaticBuffers:.*?logits: torch.Tensor\n", new_class, content, flags=re.DOTALL)

# Add pinned buffers to _ensure_static_buffers
old_ensure = r"buf = StaticBuffers\((.*?)\)\n        self\._static\[key\] = buf"
new_ensure = """buf = StaticBuffers(\\1)
        # Allocate pinned memory for staging
        buf.cpu_input_ids = torch.zeros(B, dtype=torch.int64, pin_memory=True)
        buf.cpu_position_ids = torch.zeros(B, dtype=torch.int64, pin_memory=True)
        buf.cpu_seq_lens = torch.zeros(B, dtype=torch.int32, pin_memory=True)
        buf.cpu_context_lens = torch.zeros(B, dtype=torch.int32, pin_memory=True)
        buf.cpu_slot_mapping = torch.full((B,), dummy_slot, dtype=torch.int64, pin_memory=True)
        buf.cpu_block_tables = torch.full((B, MB), self._dummy_block, dtype=torch.int32, pin_memory=True)
        self._static[key] = buf"""
content = re.sub(old_ensure, new_ensure, content, flags=re.DOTALL)

# Rewrite prepare_static_inputs
old_prepare = r"def prepare_static_inputs.*?return buf"
new_prepare = """def prepare_static_inputs(
        self,
        token_ids: list[int],
        positions: list[int],
        block_tables: list[list[int]],
        seq_lens: list[int],
        slot_mappings: list[int],
        batch_bucket: int,
        ctx_bucket: int,
    ) -> StaticBuffers:
        buf = self._ensure_static_buffers(batch_bucket, ctx_bucket)
        B_actual = len(token_ids)
        dummy_slot = self._dummy_block * self.block_size

        self._seq_lens_cpu = list(seq_lens) + [0] * (batch_bucket - B_actual)

        # 1. Fill CPU pinned buffers directly
        buf.cpu_input_ids[:B_actual] = torch.tensor(token_ids, dtype=torch.int64)
        buf.cpu_position_ids[:B_actual] = torch.tensor(positions, dtype=torch.int64)
        buf.cpu_seq_lens[:B_actual] = torch.tensor(seq_lens, dtype=torch.int32)
        buf.cpu_context_lens[:B_actual] = torch.tensor([s - 1 for s in seq_lens], dtype=torch.int32)
        buf.cpu_slot_mapping[:B_actual] = torch.tensor(slot_mappings, dtype=torch.int64)
        
        if B_actual < batch_bucket:
            buf.cpu_input_ids[B_actual:].zero_()
            buf.cpu_position_ids[B_actual:].zero_()
            buf.cpu_seq_lens[B_actual:].zero_()
            buf.cpu_context_lens[B_actual:].zero_()
            buf.cpu_slot_mapping[B_actual:].fill_(dummy_slot)
            buf.cpu_block_tables[B_actual:].fill_(self._dummy_block)

        # Batch the block tables formatting on CPU
        for i, bt in enumerate(block_tables):
            n = len(bt)
            buf.cpu_block_tables[i, :n] = torch.tensor(bt, dtype=torch.int32)
            if n < self.max_blocks_per_seq:
                buf.cpu_block_tables[i, n:].fill_(self._dummy_block)

        # 2. Issue non-blocking copies to GPU
        buf.input_ids.copy_(buf.cpu_input_ids, non_blocking=True)
        buf.position_ids.copy_(buf.cpu_position_ids, non_blocking=True)
        buf.seq_lens.copy_(buf.cpu_seq_lens, non_blocking=True)
        buf.context_lens.copy_(buf.cpu_context_lens, non_blocking=True)
        buf.slot_mapping.copy_(buf.cpu_slot_mapping, non_blocking=True)
        buf.block_tables.copy_(buf.cpu_block_tables, non_blocking=True)

        return buf"""
content = re.sub(old_prepare, new_prepare, content, flags=re.DOTALL)

with open("babyvllm/worker/model_runner.py", "w") as f:
    f.write(content)
