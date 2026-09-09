from contextlib import contextmanager

import torch
from babyvllm.layers.attention import Attention
from babyvllm.models.qwen2 import Qwen2ForCausalLM
from babyvllm.worker.context import AttentionMetadata, set_forward_context
from babyvllm.layers.sampler import Sampler


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


class ModelRunner:
    def __init__(self, model_config, cache_config=None, scheduler_config=None, device=None):
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.block_size = cache_config.block_size if cache_config else getattr(model_config, 'block_size', 16)
        self.device = torch.device(device) if device is not None else pick_device()
        self.dtype = pick_dtype(self.device)
        
        torch.set_default_dtype(self.dtype)
        with torch.device(self.device):
            self.model = Qwen2ForCausalLM(model_config)
        self.model.eval()
        torch.set_default_dtype(torch.float32)

        self.sampler = Sampler(model_config.vocab_size).to(self.device)
        self.num_blocks: int | None = None
        self.kv_profile: dict | None = None

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
        budget = int(total_memory * utilization) - allocated
        available_for_kv = min(budget, int(free_bytes) - slack)
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
        """Profile free memory, then allocate the paged KV pool once."""
        if self.num_blocks is not None:
            return self.num_blocks

        self.num_blocks = self.determine_num_blocks()
        while True:
            try:
                self._alloc_kv_tensors(self.num_blocks)
                break
            except torch.cuda.OutOfMemoryError:
                for module in self.model.modules():
                    if isinstance(module, Attention):
                        module.kv_cache = None
                torch.cuda.empty_cache()
                if self.num_blocks <= 1:
                    raise
                self.num_blocks = max(1, self.num_blocks // 2)
                if self.kv_profile is not None:
                    self.kv_profile["num_blocks"] = self.num_blocks
                    self.kv_profile["max_kv_tokens"] = self.num_blocks * self.block_size
                    self.kv_profile["oom_backoff"] = True
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

    @torch.inference_mode()
    def execute_model(self, scheduler_output) -> dict[int, int]:
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
