import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from babyvllm.layers.attention import Attention
from babyvllm.models.qwen2 import Qwen2ForCausalLM
from babyvllm.worker.context import AttentionMetadata, set_forward_context
from babyvllm.worker.decode_inputs import DecodeInputs
from babyvllm.worker.graph_runner import CUDAGraphRunner
from babyvllm.layers.sampler import Sampler


BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64]
CTX_BUCKETS = [256, 2048]
# Length of the sizing forward. The scheduler budget is a sum over sequences
# (64 × 1920 = 122_880 at the Day 6 point), not one causal sequence. Running
# that sum through math SDPA allocates a 122_880² fp32 score matrix, 56.25 GiB,
# and the LM head on every row is another ~70 GiB. Neither tensor exists in a
# real step. 128 tokens still crosses the GEMV/GEMM boundary, and a math-backend
# score matrix at that length is 64 KiB per head.
MEMORY_PROFILE_TOKENS = 128

# Greedy argmax inside the decode graph. "fp32" copies the bf16 logits to fp32
# and runs torch.argmax. "bf16" has the LM head write bf16 logits straight into
# the static buffer and runs torch.argmax on them. "triton" does the same with
# a split-row kernel. All three return the same tokens; kept switchable for A/B runs.
# Triton is the default: 0.05 ms less per batch-64 step than fp32 on the H100.
GREEDY_ARGMAX_MODES = ("fp32", "bf16", "triton")


def profile_forward_tokens(max_batched_tokens: int, max_position_embeddings: int) -> int:
    """How many tokens the KV-cache sizing forward should run."""
    return max(1, min(max_batched_tokens, max_position_embeddings, MEMORY_PROFILE_TOKENS))


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
    slack = 512 * 1024 * 1024  # 512 MB slack for staging buffers and miscellaneous allocations
    return slack


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
    # Greedy argmax of ``logits``, written inside the graph.
    tokens: torch.Tensor


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
        self._decode_inputs: DecodeInputs | None = None
        self.greedy_argmax = os.environ.get("BABYVLLM_GREEDY_ARGMAX", "triton")
        if self.greedy_argmax not in GREEDY_ARGMAX_MODES:
            raise ValueError(f"BABYVLLM_GREEDY_ARGMAX must be one of {GREEDY_ARGMAX_MODES}")
        max_ctx = max(CTX_BUCKETS)
        self.max_blocks_per_seq = (max_ctx + self.block_size - 1) // self.block_size + 1

    def _bytes_per_token(self) -> int:
        head_dim = self.model_config.head_dim
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
        profile_tokens = profile_forward_tokens(
            max_batched_tokens, self.model_config.max_position_embeddings
        )
        self._profile_activation_memory(profile_tokens)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.max_memory_allocated()
        allocated = torch.cuda.memory_allocated()
        free_bytes, total_memory = torch.cuda.mem_get_info()
        utilization = 0.9
        slack = 256 * 1024 * 1024
        pool_reserve = 0
        if self.use_cuda_graphs:
            head_dim = self.model_config.head_dim
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
            "profile_tokens": profile_tokens,
            "max_num_batched_tokens": max_batched_tokens,
        }
        return num_blocks

    def _profile_activation_memory(self, n_tokens: int) -> None:
        """Run one short forward so CUDA workspaces exist, then drop it.

        Logits are taken for the last token only, matching prefill and decode.
        A full-sequence LM head is ``n × vocab`` fp32 and is not part of the
        memory this pass is measuring.
        """
        dummy_input = torch.zeros(n_tokens, dtype=torch.int64, device=self.device)
        dummy_positions = torch.arange(n_tokens, dtype=torch.int64, device=self.device)
        with torch.inference_mode():
            hidden_states = self.model(dummy_input, dummy_positions)
            _ = self.model.compute_logits(hidden_states[-1:])
            del hidden_states

    def _alloc_kv_tensors(self, num_blocks: int) -> None:
        num_kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.head_dim
        for module in self.model.modules():
            if isinstance(module, Attention):
                # Zeroed, not empty: the torch decode path reads padded slots past
                # seq_len, and a NaN there survives the -inf mask in softmax @ V.
                module.kv_cache = torch.zeros(
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
            logits=torch.zeros(
                B, V,
                dtype=torch.float32 if self.greedy_argmax == "fp32" else self.dtype,
                device=dev,
            ),
            tokens=torch.zeros(B, dtype=torch.int64, device=dev),
        )
        self._static[key] = buf
        return buf

    def _host_decode_inputs(self) -> DecodeInputs:
        if self._decode_inputs is None:
            self._decode_inputs = DecodeInputs(
                max_batch=max(BATCH_BUCKETS),
                max_blocks=self.max_blocks_per_seq,
                block_size=self.block_size,
                dummy_block=self._dummy_block,
                pin=self.device.type == "cuda",
            )
        return self._decode_inputs

    def prepare_decode_inputs(
        self,
        seqs,
        block_tables: dict[int, list[int]],
        batch_bucket: int,
        ctx_bucket: int,
    ) -> StaticBuffers:
        """Edit the staging arrays for this step and queue the copies to the graph's inputs."""
        buf = self._ensure_static_buffers(batch_bucket, ctx_bucket)
        host = self._host_decode_inputs()
        host.update(seqs, block_tables, batch_bucket)
        B = batch_bucket
        buf.input_ids.copy_(host.t_input_ids[:B], non_blocking=True)
        buf.position_ids.copy_(host.t_positions[:B], non_blocking=True)
        buf.seq_lens.copy_(host.t_seq_lens[:B], non_blocking=True)
        buf.context_lens.copy_(host.t_context_lens[:B], non_blocking=True)
        buf.slot_mapping.copy_(host.t_slot_mapping[:B], non_blocking=True)
        buf.block_tables.copy_(host.t_block_tables[:B], non_blocking=True)
        return buf

    @torch.inference_mode()
    def _forward_static(
        self,
        buf: StaticBuffers,
        batch_bucket: int,
        ctx_bucket: int,
    ) -> torch.Tensor:
        B = batch_bucket
        # Host-side copies of the lengths are only read by the torch fallback
        # paths, which a one-token CUDA step never takes. The values that count
        # live in buf.seq_lens and buf.context_lens.
        seq_lens_cpu = [ctx_bucket] * B
        context_lens_cpu = [ctx_bucket - 1] * B
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

        # Greedy sampling inside the graph. Stochastic requests read buf.logits,
        # and the sampler upcasts them, which is exact from bf16.
        if self.greedy_argmax == "fp32":
            buf.logits[:B].copy_(self.model.compute_logits(hidden))
            buf.tokens[:B].copy_(torch.argmax(buf.logits[:B], dim=-1))
        else:
            self.model.compute_logits(hidden, out=buf.logits[:B])
            if self.greedy_argmax == "triton":
                from babyvllm.kernels.argmax import greedy_argmax
                greedy_argmax(buf.logits[:B], buf.tokens[:B])
            else:
                buf.tokens[:B].copy_(torch.argmax(buf.logits[:B], dim=-1))
        return buf.logits

    def capture_one(self, batch_bucket: int, ctx_bucket: int, warmup_steps: int = 3):
        key = (batch_bucket, ctx_bucket)
        if key in self._graphs:
            return self._graphs[key]

        buf = self._ensure_static_buffers(batch_bucket, ctx_bucket)
        side_stream = torch.cuda.Stream()
        # The static buffers were just written on the current stream.
        side_stream.wait_stream(torch.cuda.current_stream())
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
        n = len(seqs)
        bb = ceil_to_bucket(n, BATCH_BUCKETS)
        cb = ceil_to_bucket(max(s.num_computed_tokens for s in seqs) + 1, CTX_BUCKETS)
        assert bb is not None and cb is not None

        with nvtx_range("prepare_inputs"):
            buf = self.prepare_decode_inputs(seqs, scheduler_output.block_tables, bb, cb)

        with nvtx_range("forward"):
            self.capture_one(bb, cb)
            self.replay(buf, bb, cb)

        with nvtx_range("sample"):
            if all(s.sampling_params.temperature == 0.0 for s in seqs):
                # Argmax already ran inside the graph. One 8-byte-per-row copy
                # to pinned memory, one stream sync, no extra kernels.
                host = self._decode_inputs
                host.t_tokens[:bb].copy_(buf.tokens[:bb], non_blocking=True)
                torch.cuda.current_stream().synchronize()
                tokens = host.tokens[:n].tolist()
                return {s.seq_id: t for s, t in zip(seqs, tokens)}
            return self.sampler(buf.logits[:n], None, scheduler_output)

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
