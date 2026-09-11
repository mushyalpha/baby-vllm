from contextlib import contextmanager
from dataclasses import dataclass

import torch
from babyvllm.layers.attention import Attention
from babyvllm.models.qwen2 import Qwen2ForCausalLM
from babyvllm.worker.context import AttentionMetadata, set_forward_context
from babyvllm.worker.graph_runner import CUDAGraphRunner
from babyvllm.layers.sampler import Sampler


BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64]
CTX_BUCKETS = [256, 2048]


@contextmanager
def nvtx_range(name: str):
    enabled = torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def ceil_to_bucket(value: int, buckets: list[int]) -> int | None:
    for b in buckets:
        if value <= b:
            return b
    return None


def compute_pool_reserve(
    max_batch_bucket: int,
    max_ctx_bucket: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    bytes_per_element: int = 2,
) -> int:
    gather_per_layer = (
        max_batch_bucket * n_kv_heads * max_ctx_bucket * head_dim
        * 2
        * bytes_per_element
    )
    total_gather = gather_per_layer * n_layers
    slack = 2 * 1024 * 1024 * 1024
    return total_gather + slack


@dataclass
class StaticBuffers:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    logits: torch.Tensor


class ModelRunner:
    def __init__(
        self,
        model_config,
        cache_config=None,
        scheduler_config=None,
        device=None,
        use_cuda_graphs: bool | None = None,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.block_size = cache_config.block_size if cache_config else getattr(model_config, 'block_size', 16)
        self.device = torch.device(device) if device is not None else pick_device()
        self.dtype = pick_dtype(self.device)
        if use_cuda_graphs is None:
            self.use_cuda_graphs = self.device.type == "cuda"
        else:
            self.use_cuda_graphs = bool(use_cuda_graphs) and self.device.type == "cuda"

        torch.set_default_dtype(self.dtype)
        with torch.device(self.device):
            self.model = Qwen2ForCausalLM(model_config)
        self.model.eval()
        torch.set_default_dtype(torch.float32)

        self.sampler = Sampler(model_config.vocab_size).to(self.device)
        self.num_blocks: int | None = None
        self.kv_profile: dict | None = None
        self._dummy_block = 0
        self._static: dict[tuple[int, int], StaticBuffers] = {}
        self._graphs: dict[tuple[int, int], CUDAGraphRunner] = {}
        self._pool = None
        self._seq_lens_cpu: list[int] = []
        max_ctx = max(CTX_BUCKETS)
        self.max_blocks_per_seq = (max_ctx + self.block_size - 1) // self.block_size + 1

    def _bytes_per_token(self) -> int:
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
        bytes_per_element = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        return (
            2
            * self.model_config.num_key_value_heads
            * head_dim
            * self.model_config.num_hidden_layers
            * bytes_per_element
        )

    def determine_num_blocks(self) -> int:
        if self.device.type != "cuda":
            num_blocks = self.cache_config.num_cpu_blocks if self.cache_config else 100
            self.kv_profile = {
                "num_blocks": num_blocks,
                "block_size": self.block_size,
                "bytes_per_token": self._bytes_per_token(),
            }
            return num_blocks

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        max_batched_tokens = (
            self.scheduler_config.max_num_batched_tokens if self.scheduler_config else 2048
        )
        dummy_input = torch.zeros(max_batched_tokens, dtype=torch.int64, device=self.device)
        dummy_positions = torch.arange(max_batched_tokens, dtype=torch.int64, device=self.device)

        with torch.inference_mode():
            hidden_states = self.model(dummy_input, dummy_positions)
            _ = self.model.compute_logits(hidden_states)
            del hidden_states

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.max_memory_allocated()
        allocated = torch.cuda.memory_allocated()
        free_bytes, total_memory = torch.cuda.mem_get_info()
        utilization = 0.9
        slack = 256 * 1024 * 1024
        pool_reserve = 0
        if self.use_cuda_graphs:
            head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
            pool_reserve = compute_pool_reserve(
                max(BATCH_BUCKETS),
                max(CTX_BUCKETS),
                self.model_config.num_key_value_heads,
                head_dim,
                self.model_config.num_hidden_layers,
                2 if self.dtype in (torch.float16, torch.bfloat16) else 4,
            )
        budget = int(total_memory * utilization) - allocated
        available_for_kv = min(budget, int(free_bytes) - slack - pool_reserve)
        bytes_per_token = self._bytes_per_token()
        bytes_per_block = bytes_per_token * self.block_size
        num_blocks = max(1, int(available_for_kv / bytes_per_block)) if available_for_kv > 0 else 1

        self.kv_profile = {
            "peak_memory_gb": peak_memory / 1e9,
            "allocated_gb": allocated / 1e9,
            "free_memory_gb": free_bytes / 1e9,
            "total_memory_gb": total_memory / 1e9,
            "utilization": utilization,
            "available_for_kv_gb": max(0.0, available_for_kv) / 1e9,
            "bytes_per_token": bytes_per_token,
            "block_size": self.block_size,
            "num_blocks": num_blocks,
            "max_kv_tokens": num_blocks * self.block_size,
        }
        return num_blocks

    def _alloc_kv_tensors(self, num_blocks: int) -> None:
        num_kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache = torch.empty(
                    2,
                    num_blocks,
                    self.block_size,
                    num_kv_heads,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )

    def allocate_kv_cache(self) -> int:
        if self.num_blocks is not None:
            return self.num_blocks

        usable = self.determine_num_blocks()
        alloc = usable + 1 if self.use_cuda_graphs else usable
        while True:
            try:
                self._alloc_kv_tensors(alloc)
                break
            except torch.cuda.OutOfMemoryError:
                for module in self.model.modules():
                    if isinstance(module, Attention):
                        module.kv_cache = None
                torch.cuda.empty_cache()
                if alloc <= 1:
                    raise
                alloc = max(1, alloc // 2)
                if self.kv_profile is not None:
                    usable_now = alloc - 1 if self.use_cuda_graphs and alloc > 1 else alloc
                    self.kv_profile["num_blocks"] = usable_now
                    self.kv_profile["max_kv_tokens"] = usable_now * self.block_size
                    self.kv_profile["oom_backoff"] = True

        if self.use_cuda_graphs and alloc > 1:
            self._dummy_block = alloc - 1
            self.num_blocks = alloc - 1
        else:
            self._dummy_block = 0
            self.num_blocks = alloc
        if self.kv_profile is not None:
            self.kv_profile["num_blocks"] = self.num_blocks
            self.kv_profile["max_kv_tokens"] = self.num_blocks * self.block_size
        return self.num_blocks

    def prepare_inputs(self, scheduler_output) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        slot_mapping = []
        seq_lens = []
        context_lens = []
        block_tables = []
        query_lens = []
        input_tokens = []
        positions_list = []
        
        max_query_len = 0
        max_seq_len = 0
        max_num_blocks = 0

        for seq in scheduler_output.scheduled_sequences:
            needed = scheduler_output.num_scheduled_tokens[seq.seq_id]
            start_idx = seq.num_computed_tokens
            
            query_tokens = seq.token_ids[start_idx : start_idx + needed]
            input_tokens.extend(query_tokens)
            query_lens.append(needed)
            
            ctx_len = seq.num_computed_tokens
            context_lens.append(ctx_len)
            
            seq_len = ctx_len + needed
            seq_lens.append(seq_len)
            
            max_query_len = max(max_query_len, needed)
            max_seq_len = max(max_seq_len, seq_len)
            
            b_table = scheduler_output.block_tables[seq.seq_id]
            block_tables.append(b_table)
            max_num_blocks = max(max_num_blocks, len(b_table))
            
            s_map = scheduler_output.slot_mappings[seq.seq_id]
            slot_mapping.extend(s_map)
            
            positions_list.extend(range(ctx_len, ctx_len + needed))
            
        padded_block_tables = []
        for bt in block_tables:
            padded_block_tables.append(bt + [0] * (max_num_blocks - len(bt)))

        query_start_loc = [0]
        curr = 0
        for qlen in query_lens:
            curr += qlen
            query_start_loc.append(curr)

        attn_metadata = AttentionMetadata(
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64, device=self.device),
            query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=self.device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_tables=torch.tensor(padded_block_tables, dtype=torch.int32, device=self.device),
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens_cpu=seq_lens,
            context_lens_cpu=context_lens,
            query_start_loc_cpu=query_start_loc,
        )
        
        flat_input_ids = torch.tensor(input_tokens, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
        return flat_input_ids, positions, attn_metadata

    def _can_graph_decode(self, scheduler_output) -> bool:
        if not self.use_cuda_graphs:
            return False
        seqs = scheduler_output.scheduled_sequences
        if not seqs or self.num_blocks is None:
            return False
        for seq in seqs:
            if scheduler_output.num_scheduled_tokens[seq.seq_id] != 1:
                return False
        if ceil_to_bucket(len(seqs), BATCH_BUCKETS) is None:
            return False
        max_seq = 0
        for seq in seqs:
            max_seq = max(max_seq, seq.num_computed_tokens + 1)
        return ceil_to_bucket(max_seq, CTX_BUCKETS) is not None

    def _ensure_static_buffers(self, batch_bucket: int, ctx_bucket: int) -> StaticBuffers:
        key = (batch_bucket, ctx_bucket)
        if key in self._static:
            return self._static[key]

        B = batch_bucket
        MB = self.max_blocks_per_seq
        V = self.model_config.vocab_size
        dev = self.device
        dummy_slot = self._dummy_block * self.block_size
        buf = StaticBuffers(
            input_ids=torch.zeros(B, dtype=torch.int64, device=dev),
            position_ids=torch.zeros(B, dtype=torch.int64, device=dev),
            block_tables=torch.full((B, MB), self._dummy_block, dtype=torch.int32, device=dev),
            seq_lens=torch.zeros(B, dtype=torch.int32, device=dev),
            context_lens=torch.zeros(B, dtype=torch.int32, device=dev),
            slot_mapping=torch.full((B,), dummy_slot, dtype=torch.int64, device=dev),
            query_start_loc=torch.arange(B + 1, dtype=torch.int32, device=dev),
            logits=torch.zeros(B, V, dtype=torch.float32, device=dev),
        )
        self._static[key] = buf
        return buf

    def prepare_static_inputs(
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

        buf.input_ids[:B_actual].copy_(
            torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        )
        buf.position_ids[:B_actual].copy_(
            torch.tensor(positions, dtype=torch.int64, device=self.device)
        )
        buf.seq_lens[:B_actual].copy_(
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        )
        buf.context_lens[:B_actual].copy_(
            torch.tensor([s - 1 for s in seq_lens], dtype=torch.int32, device=self.device)
        )
        buf.slot_mapping[:B_actual].copy_(
            torch.tensor(slot_mappings, dtype=torch.int64, device=self.device)
        )

        if B_actual < batch_bucket:
            buf.input_ids[B_actual:].zero_()
            buf.position_ids[B_actual:].zero_()
            buf.seq_lens[B_actual:].zero_()
            buf.context_lens[B_actual:].zero_()
            buf.slot_mapping[B_actual:].fill_(dummy_slot)
            buf.block_tables[B_actual:].fill_(self._dummy_block)

        for i, bt in enumerate(block_tables):
            n = len(bt)
            buf.block_tables[i, :n].copy_(
                torch.tensor(bt, dtype=torch.int32, device=self.device)
            )
            if n < self.max_blocks_per_seq:
                buf.block_tables[i, n:].fill_(self._dummy_block)

        return buf

    @torch.inference_mode()
    def _forward_static(
        self,
        buf: StaticBuffers,
        batch_bucket: int,
        ctx_bucket: int,
    ) -> torch.Tensor:
        B = batch_bucket
        seq_lens_cpu = (
            self._seq_lens_cpu[:B] if len(self._seq_lens_cpu) >= B else [0] * B
        )
        context_lens_cpu = [max(0, s - 1) for s in seq_lens_cpu]
        attn_metadata = AttentionMetadata(
            slot_mapping=buf.slot_mapping[:B],
            block_tables=buf.block_tables[:B],
            query_start_loc=buf.query_start_loc[: B + 1],
            seq_lens=buf.seq_lens[:B],
            context_lens=buf.context_lens[:B],
            max_query_len=1,
            max_seq_len=ctx_bucket,
            query_start_loc_cpu=list(range(B + 1)),
            seq_lens_cpu=seq_lens_cpu,
            context_lens_cpu=context_lens_cpu,
        )

        with set_forward_context(attn_metadata):
            hidden = self.model(buf.input_ids[:B], buf.position_ids[:B])

        logits = self.model.compute_logits(hidden)
        buf.logits[:B].copy_(logits)
        return buf.logits

    def capture_one(self, batch_bucket: int, ctx_bucket: int, warmup_steps: int = 3):
        key = (batch_bucket, ctx_bucket)
        if key in self._graphs:
            return self._graphs[key]

        buf = self._ensure_static_buffers(batch_bucket, ctx_bucket)
        if len(self._seq_lens_cpu) < batch_bucket:
            self._seq_lens_cpu = [1] * batch_bucket

        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(warmup_steps):
                self._forward_static(buf, batch_bucket, ctx_bucket)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        runner = CUDAGraphRunner()
        runner.capture(
            self._forward_static, buf, batch_bucket, ctx_bucket, pool=self._pool
        )
        self._graphs[key] = runner
        if self._pool is None:
            self._pool = runner.pool
        return runner

    def capture_all(self, warmup_steps: int = 3):
        for cb in reversed(CTX_BUCKETS):
            for bb in reversed(BATCH_BUCKETS):
                self.capture_one(bb, cb, warmup_steps=warmup_steps)

    def replay(self, buf: StaticBuffers, batch_bucket: int, ctx_bucket: int) -> torch.Tensor:
        key = (batch_bucket, ctx_bucket)
        if key not in self._graphs:
            raise RuntimeError(f"No graph captured for {key}. Call capture_one() first.")
        self._graphs[key].replay()
        return buf.logits

    def _execute_decode_graph(self, scheduler_output) -> dict[int, int]:
        seqs = scheduler_output.scheduled_sequences
        token_ids = []
        positions = []
        block_tables = []
        seq_lens = []
        slot_mappings = []
        for seq in seqs:
            start = seq.num_computed_tokens
            token_ids.append(seq.token_ids[start])
            positions.append(start)
            seq_lens.append(start + 1)
            block_tables.append(scheduler_output.block_tables[seq.seq_id])
            slot_mappings.append(scheduler_output.slot_mappings[seq.seq_id][0])

        bb = ceil_to_bucket(len(seqs), BATCH_BUCKETS)
        cb = ceil_to_bucket(max(seq_lens), CTX_BUCKETS)
        assert bb is not None and cb is not None

        with nvtx_range("prepare_inputs"):
            buf = self.prepare_static_inputs(
                token_ids, positions, block_tables, seq_lens, slot_mappings, bb, cb
            )

        with nvtx_range("forward"):
            self.capture_one(bb, cb)
            self.replay(buf, bb, cb)

        with nvtx_range("sample"):
            sampled_tokens = self.sampler(
                buf.logits[: len(seqs)], None, scheduler_output
            )
        return sampled_tokens

    @torch.inference_mode()
    def _execute_eager(self, scheduler_output) -> dict[int, int]:
        with nvtx_range("prepare_inputs"):
            input_ids, positions, attn_metadata = self.prepare_inputs(scheduler_output)

        with nvtx_range("forward"), set_forward_context(attn_metadata):
            hidden_states = self.model(input_ids, positions)

        last_indices = [end - 1 for end in attn_metadata.query_start_loc_cpu[1:]]
        hidden_states = hidden_states[last_indices]

        with nvtx_range("sample"):
            logits = self.model.compute_logits(hidden_states)
            sampled_tokens = self.sampler(logits, attn_metadata, scheduler_output)
        return sampled_tokens

    @torch.inference_mode()
    def execute_model(self, scheduler_output) -> dict[int, int]:
        if self._can_graph_decode(scheduler_output):
            #decode
            return self._execute_decode_graph(scheduler_output)
        return self._execute_eager(scheduler_output)
