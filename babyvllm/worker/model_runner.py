import torch
from babyvllm.layers.attention import Attention
from babyvllm.models.qwen2 import Qwen2ForCausalLM
from babyvllm.worker.context import AttentionMetadata, set_forward_context
from babyvllm.layers.sampler import Sampler


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
        
        self.num_blocks = self.determine_num_blocks()
        
        num_kv_heads = model_config.num_key_value_heads
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache = torch.empty(2, self.num_blocks, self.block_size, num_kv_heads, head_dim, dtype=self.dtype, device=self.device)
        
    def determine_num_blocks(self) -> int:
        if self.device.type != "cuda":
            return self.cache_config.num_cpu_blocks if self.cache_config else 100
            
        torch.cuda.reset_peak_memory_stats()
        
        max_batched_tokens = self.scheduler_config.max_num_batched_tokens if self.scheduler_config else 2048
        
        dummy_input = torch.zeros(max_batched_tokens, dtype=torch.int64, device=self.device)
        dummy_positions = torch.arange(max_batched_tokens, dtype=torch.int64, device=self.device)
        
        with torch.inference_mode():
            hidden_states = self.model(dummy_input, dummy_positions)
            _ = self.model.compute_logits(hidden_states)
            
        peak_memory = torch.cuda.max_memory_allocated()
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        
        utilization = 0.9
        available_for_kv = total_memory * utilization - peak_memory
        
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
        bytes_per_element = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        
        bytes_per_token = 2 * self.model_config.num_key_value_heads * head_dim * self.model_config.num_hidden_layers * bytes_per_element
        bytes_per_block = bytes_per_token * self.block_size
        
        num_blocks = int(available_for_kv / bytes_per_block)
        return max(1, num_blocks)

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
        input_ids, positions, attn_metadata = self.prepare_inputs(scheduler_output)
        
        with set_forward_context(attn_metadata):
            hidden_states = self.model(input_ids, positions)
            
        last_indices = [end - 1 for end in attn_metadata.query_start_loc_cpu[1:]]
        hidden_states = hidden_states[last_indices]
            
        logits = self.model.compute_logits(hidden_states)
        sampled_tokens = self.sampler(logits, attn_metadata, scheduler_output)
        return sampled_tokens
