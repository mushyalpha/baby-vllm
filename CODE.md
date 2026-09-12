sequence.py

```python
import itertools
from enum import Enum, auto
from dataclasses import dataclass, field

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_tokens: int = 0
    seed: int | None = None
    ignore_eos: bool = False
    max_tokens: int = 256
    stop_tokens: set[int] = field(default_factory=set)


class SequenceState(Enum):
    WAITING = auto()
    RUNNING = auto()
    DONE = auto()

class SequenceFinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"

class Sequence:
    _counter = itertools.count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        if not token_ids:
            raise ValueError("Prompt cannot be empty")
        self.seq_id = next(Sequence._counter)
        self.token_ids = list(token_ids)
        self.sampling_params = sampling_params or SamplingParams()
        self.status = SequenceState.WAITING
        self.finish_reason = None
        self._prompt_len = len(token_ids)
        self.num_computed_tokens: int = 0


    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, item):
        return self.token_ids[item]

    def __repr__(self) -> str:
        return (f"Sequence(id={self.seq_id}, {self.status.name}, "
                f"prompt={self._prompt_len}, total={len(self)}, "
                f"computed={self.num_computed_tokens})")

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self._prompt_len:]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def advance_computed(self, n: int) -> None:
        self.num_computed_tokens += n
        if self.num_computed_tokens > len(self.token_ids):
            raise ValueError(f"Computed tokens ({self.num_computed_tokens}) exceeds sequence length ({len(self.token_ids)})")

    @property
    def prompt_len(self) -> int:
        return self._prompt_len

    @property
    def generated_token_len(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def num_tokens_to_compute(self) -> int:
        return len(self.token_ids) - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceState.DONE

    @property
    def last_token_id(self):
        return self.token_ids[-1]

    def finish(self, reason: SequenceFinishReason) -> None:
        self.status = SequenceState.DONE
        self.finish_reason = reason

    def reset_for_recompute(self) -> None:
        self.status = SequenceState.WAITING
        self.num_computed_tokens = 0
```

kv_cache_manager.py

```python
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
```

config.py

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    block_size: int = 16
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int = 100
    num_cpu_blocks: int = 100

@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
```

scheduler.py

```python
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
    def __init__(self, kv_cache_manager: KVCacheManager, max_num_batched_tokens: int = 8, max_num_seqs: int = 4):
        
        
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

    def update_sequence(self, seq: Sequence):
        if seq.generated_token_len > 0:
            token_id = seq.last_token_id
            if token_id in seq.sampling_params.stop_tokens:
                self._finish(seq, SequenceFinishReason.STOP)
            elif seq.generated_token_len >= seq.sampling_params.max_tokens:
                self._finish(seq, SequenceFinishReason.LENGTH)

    def update_from_output(self, scheduler_output: SchedulerOutput, model_output: dict[int, int]):
        for seq in scheduler_output.scheduled_sequences:
            num_scheduled = scheduler_output.num_scheduled_tokens[seq.seq_id]
            seq.advance_computed(num_scheduled)
            if seq.seq_id in model_output:
                seq.append_token(model_output[seq.seq_id])
                self.update_sequence(seq)

    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        scheduled_sequences: list[Sequence] = []
        num_scheduled_tokens: dict[int, int] = {}
        block_tables: dict[int, list[int]] = {}
        slot_mappings: dict[int, list[int]] = {}

        running_seqs = list(self.running)
        preempted_seqs = []
        for seq in running_seqs:
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
```

engine.py

```python
from dataclasses import dataclass

from babyvllm.sequence import (
    Sequence,
    SamplingParams,
    SequenceFinishReason,
)
from babyvllm.scheduler import Scheduler, SchedulerOutput
from babyvllm.kv_cache_manager import KVCacheManager


@dataclass
class RequestOutput:
    seq_id: int
    new_token_ids: list[int]
    finish_reason: SequenceFinishReason | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class LLMEngine:
    def __init__(
        self,
        model_runner,
        max_num_batched_tokens: int = 8,
        max_num_seqs: int = 4,
    ):
        self.model_runner = model_runner

        self.kv_cache_manager = KVCacheManager(
            num_blocks=model_runner.determine_num_blocks(),
            block_size=model_runner.block_size,
        )

        self.scheduler = Scheduler(
            kv_cache_manager=self.kv_cache_manager,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )

        self.sequences: dict[int, Sequence] = {}

        self.block_size = model_runner.block_size

    def add_request(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> int:

        seq = Sequence(
            token_ids=token_ids,
            sampling_params=sampling_params,
        )

        self.sequences[seq.seq_id] = seq
        self.scheduler.add_seq(seq)

        return seq.seq_id

    def abort_request(self, seq_id: int) -> None:
        seq = self.sequences[seq_id]

        if not seq.is_finished:
            self.scheduler.abort_seq(seq)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> list[RequestOutput]:
        if not self.has_unfinished_requests() and not self.scheduler.finished_seqs:
            return []

        scheduler_output = None
        if self.has_unfinished_requests():
            scheduler_output = self.scheduler.schedule()

            made_progress = (
                bool(scheduler_output.scheduled_sequences) or 
                bool(self.scheduler.finished_seqs) or 
                bool(scheduler_output.preempted_seqs)
            )

            if not made_progress:
                raise RuntimeError(
                    "Engine stalled: unfinished requests exist, "
                    "but scheduler scheduled no work, preempted nothing, and no sequences finished."
                )

            if scheduler_output.scheduled_sequences:
                sampled_tokens = self.model_runner.execute_model(
                    scheduler_output
                )

                if not isinstance(sampled_tokens, dict):
                    raise RuntimeError(
                        "ModelRunner returned incorrect type"
                    )

                self.scheduler.update_from_output(
                    scheduler_output,
                    sampled_tokens,
                )

        outputs = []
        handled_seq_ids = set()

        sampled_tokens = sampled_tokens if 'sampled_tokens' in locals() else {}
        
        if scheduler_output and scheduler_output.scheduled_sequences:
            for seq in scheduler_output.scheduled_sequences:
                if seq.seq_id in sampled_tokens:
                    outputs.append(
                        RequestOutput(
                            seq_id=seq.seq_id,
                            new_token_ids=[sampled_tokens[seq.seq_id]],
                            finish_reason=seq.finish_reason,
                        )
                    )
                    handled_seq_ids.add(seq.seq_id)

        for seq in self.scheduler.pop_finished():
            if seq.seq_id not in handled_seq_ids:
                outputs.append(
                    RequestOutput(
                        seq_id=seq.seq_id,
                        new_token_ids=[],
                        finish_reason=seq.finish_reason,
                    )
                )
            self.sequences.pop(seq.seq_id, None)

        return outputs

    def run(self, max_steps: int = 1000) -> dict[int, RequestOutput]:
        steps = 0
        final_outputs: dict[int, RequestOutput] = {}
        
        while self.has_unfinished_requests():
            step_outputs = self.step()
            for out in step_outputs:
                if out.seq_id not in final_outputs:
                    final_outputs[out.seq_id] = RequestOutput(
                        seq_id=out.seq_id,
                        new_token_ids=[],
                        finish_reason=None
                    )
                final_outputs[out.seq_id].new_token_ids.extend(out.new_token_ids)
                if out.finished:
                    final_outputs[out.seq_id].finish_reason = out.finish_reason
            
            steps += 1
            if steps >= max_steps:
                raise RuntimeError(f"Engine.run() hit max_steps ({max_steps}). Possible livelock.")

        return final_outputs

    def reset(self) -> None:
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.scheduler.finished_seqs.clear()
        self.kv_cache_manager.reset()
        self.sequences.clear()
```

llm.py

```python
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from babyvllm.config import ModelConfig, CacheConfig, SchedulerConfig
from babyvllm.engine import LLMEngine
from babyvllm.worker.model_runner import ModelRunner
from babyvllm.worker.loader import load_model
from babyvllm.sequence import SamplingParams

class LLM:
    def __init__(
        self,
        model_name: str,
        cache_config: CacheConfig = None,
        scheduler_config: SchedulerConfig = None
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model_config = ModelConfig()
        
        path = snapshot_download(repo_id=model_name)
        
        self.model_runner = ModelRunner(self.model_config)
        load_model(self.model_runner.model, path)
        
        sched_cfg = scheduler_config or SchedulerConfig()
        
        self.engine = LLMEngine(
            model_runner=self.model_runner,
            max_num_batched_tokens=sched_cfg.max_num_batched_tokens,
            max_num_seqs=sched_cfg.max_num_seqs
        )

    def generate(self, prompts: list[str], sampling_params=None):
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0)

        for prompt in prompts:
            token_ids = self.tokenizer(prompt).input_ids
            self.engine.add_request(token_ids, sampling_params)
            
        final_outputs = {}
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for out in step_outputs:
                if out.seq_id not in final_outputs:
                    from babyvllm.engine import RequestOutput
                    final_outputs[out.seq_id] = RequestOutput(
                        seq_id=out.seq_id,
                        new_token_ids=[],
                        finish_reason=None
                    )
                final_outputs[out.seq_id].new_token_ids.extend(out.new_token_ids)
                if out.finished:
                    final_outputs[out.seq_id].finish_reason = out.finish_reason
        
        outputs = list(final_outputs.values())
        for out in outputs:
            out.text = self.tokenizer.decode(out.new_token_ids)
            
        return outputs
```

attention.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from babyvllm.worker.context import get_forward_context

def store_kvcache(key: torch.Tensor,
                  value: torch.Tensor,
                  kv_cache: torch.Tensor,
                  slot_mapping: torch.Tensor) -> None:
    _, _, _, num_kv_heads, head_dim = kv_cache.shape
    k_flat = kv_cache[0].view(-1, num_kv_heads, head_dim)
    v_flat = kv_cache[1].view(-1, num_kv_heads, head_dim)
    k_flat.index_copy_(0, slot_mapping, key.to(k_flat.dtype))
    v_flat.index_copy_(0, slot_mapping, value.to(v_flat.dtype))


class Attention(nn.Module):
    """Paged attention over a flattened (varlen) token batch."""

    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int,
                 scale: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.kv_cache: torch.Tensor | None = None

    def forward(self, q, k, v):
        ctx = get_forward_context()
        if self.kv_cache is None or ctx is None:
            return _naive_causal_attention(q, k, v, self.scale,
                                           self.num_queries_per_kv)
        md = ctx.attn_metadata
        store_kvcache(k, v, self.kv_cache, md.slot_mapping)
        return _paged_attention_torch(q, self.kv_cache, md, self.scale,
                                      self.num_queries_per_kv)


def _paged_attention_torch(query, kv_cache, md, scale, num_queries_per_kv):
    block_size = kv_cache.shape[2]
    num_kv_heads, head_dim = kv_cache.shape[3], kv_cache.shape[4]
    out = torch.empty_like(query)

    for i, (start, end) in enumerate(zip(md.query_start_loc_cpu[:-1],
                                         md.query_start_loc_cpu[1:])):
        seq_len = md.seq_lens_cpu[i]
        ctx_len = md.context_lens_cpu[i]
        q_len = end - start

        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = md.block_tables[i, :num_blocks]

        k_i = kv_cache[0][blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v_i = kv_cache[1][blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        if num_queries_per_kv > 1:
            k_i = k_i.repeat_interleave(num_queries_per_kv, dim=1)
            v_i = v_i.repeat_interleave(num_queries_per_kv, dim=1)

        q_i = query[start:end].transpose(0, 1)
        k_i = k_i.transpose(0, 1)
        v_i = v_i.transpose(0, 1)

        if ctx_len == 0 and q_len == seq_len:
            o = F.scaled_dot_product_attention(q_i, k_i, v_i,
                                               is_causal=True, scale=scale)
        else:
            mask = torch.ones(q_len, seq_len, dtype=torch.bool, device=query.device)
            mask = torch.tril(mask, diagonal=ctx_len)
            o = F.scaled_dot_product_attention(q_i, k_i, v_i,
                                               attn_mask=mask, scale=scale)
        out[start:end] = o.transpose(0, 1)
    return out


def _naive_causal_attention(q, k, v, scale, num_queries_per_kv):
    """Single-sequence, no-cache reference. Only for parity tests."""
    if num_queries_per_kv > 1:
        k = k.repeat_interleave(num_queries_per_kv, dim=1)
        v = v.repeat_interleave(num_queries_per_kv, dim=1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                                       v.transpose(0, 1), is_causal=True, scale=scale)
    return o.transpose(0, 1)
```

layernorm.py

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)
```

rotary.py

```python
import torch
import torch.nn as nn

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, positions: torch.Tensor):
        cos = self.cos_cached[0, 0, positions]
        sin = self.sin_cached[0, 0, positions]
        return cos, sin
```

sampler.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from babyvllm.worker.context import AttentionMetadata

class Sampler(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        scheduler_output,
    ) -> dict[int, int]:
        sampled_tokens = {}
        for i, seq in enumerate(scheduler_output.scheduled_sequences):
            params = seq.sampling_params
            logit = logits[i]
            
            if params is None or params.temperature == 0.0:
                token = torch.argmax(logit).item()
            else:
                probs = F.softmax(logit / params.temperature, dim=-1)
                if params.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > params.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum()
                token = torch.multinomial(probs, num_samples=1).item()
            sampled_tokens[seq.seq_id] = token
            
        return sampled_tokens
```

qwen2.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from babyvllm.layers.layernorm import RMSNorm
from babyvllm.layers.rotary import RotaryEmbedding, apply_rope
from babyvllm.layers.attention import Attention

class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen2Attention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads)

    def forward(self, x, cos, sin):
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        
        q, k = apply_rope(q, k, cos, sin)
        
        o = self.attn(q, k, v)
        return self.o_proj(o.reshape(-1, self.num_heads * self.head_dim))

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(cfg, layer_idx)
        self.mlp = Qwen2MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

class Qwen2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            cfg.hidden_size // cfg.num_attention_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            base=cfg.rope_theta
        )

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = Qwen2Model(cfg)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def compute_logits(self, hidden):
        w = self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, w)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)
```

context.py

```python
from dataclasses import dataclass
from contextlib import contextmanager
import torch

@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    query_start_loc_cpu: list[int]
    seq_lens_cpu: list[int]
    context_lens_cpu: list[int]
    max_query_len: int
    max_seq_len: int

@dataclass
class ForwardContext:
    attn_metadata: AttentionMetadata

_FORWARD_CTX: ForwardContext | None = None

@contextmanager
def set_forward_context(attn_metadata):
    global _FORWARD_CTX
    prev = _FORWARD_CTX
    _FORWARD_CTX = ForwardContext(attn_metadata)
    try:
        yield
    finally:
        _FORWARD_CTX = prev

def get_forward_context() -> ForwardContext | None:
    return _FORWARD_CTX
```

loader.py

```python
import os, glob, torch
from safetensors import safe_open

def load_model(model: torch.nn.Module, path: str):
    params = dict(model.named_parameters())
    loaded = set()
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as fp:
            for name in fp.keys():
                if name == "lm_head.weight" and model.lm_head is None:
                    continue
                if name not in params:
                    raise KeyError(f"unexpected checkpoint key: {name}")
                p = params[name]
                w = fp.get_tensor(name)
                assert p.shape == w.shape, (name, p.shape, w.shape)
                p.data.copy_(w.to(p.dtype))
                loaded.add(name)
    missing = set(params) - loaded
    if missing:
        raise RuntimeError(f"uninitialised params: {sorted(missing)}")
```

model_runner.py

```python
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

```

test_attention.py

```python
import torch
import torch.nn.functional as F

from babyvllm.layers.attention import Attention, _naive_causal_attention, store_kvcache
from babyvllm.worker.context import AttentionMetadata, set_forward_context

def test_prefill_parity():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    q_len = 10
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    
    q = torch.randn(q_len, num_heads, head_dim)
    k = torch.randn(q_len, num_kv_heads, head_dim)
    v = torch.randn(q_len, num_kv_heads, head_dim)
    
    expected = attn(q, k, v)
    
    block_size = 4
    num_blocks = (q_len + block_size - 1) // block_size
    kv_cache = torch.empty(2, num_blocks, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    md = AttentionMetadata(
        slot_mapping=torch.arange(q_len),
        block_tables=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens=torch.tensor([q_len], dtype=torch.int32),
        context_lens=torch.tensor([0], dtype=torch.int32),
        query_start_loc_cpu=[0, q_len],
        seq_lens_cpu=[q_len],
        context_lens_cpu=[0],
        max_query_len=q_len,
        max_seq_len=q_len,
    )
    
    with set_forward_context(md):
        actual = attn(q, k, v)
        
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

def test_prefill_then_decode_parity():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    
    q_all = torch.randn(11, num_heads, head_dim)
    k_all = torch.randn(11, num_kv_heads, head_dim)
    v_all = torch.randn(11, num_kv_heads, head_dim)
    
    expected = _naive_causal_attention(q_all, k_all, v_all, attn.scale, attn.num_queries_per_kv)
    
    block_size = 4
    kv_cache = torch.empty(2, 3, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    store_kvcache(k_all[:10], v_all[:10], kv_cache, torch.arange(10))
    
    q_decode = q_all[10:11]
    k_decode = k_all[10:11]
    v_decode = v_all[10:11]
    
    md = AttentionMetadata(
        slot_mapping=torch.tensor([10]),
        block_tables=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([11], dtype=torch.int32),
        context_lens=torch.tensor([10], dtype=torch.int32),
        query_start_loc_cpu=[0, 1],
        seq_lens_cpu=[11],
        context_lens_cpu=[10],
        max_query_len=1,
        max_seq_len=11,
    )
    
    with set_forward_context(md):
        actual_decode = attn(q_decode, k_decode, v_decode)
        
    torch.testing.assert_close(actual_decode[0], expected[10], atol=1e-5, rtol=1e-5)


def test_batched_decode():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 4

    attn = Attention(num_heads, head_dim, num_kv_heads)

    q0 = torch.randn(7, num_heads, head_dim)
    k0 = torch.randn(7, num_kv_heads, head_dim)
    v0 = torch.randn(7, num_kv_heads, head_dim)
    q1 = torch.randn(6, num_heads, head_dim)
    k1 = torch.randn(6, num_kv_heads, head_dim)
    v1 = torch.randn(6, num_kv_heads, head_dim)

    exp0 = _naive_causal_attention(q0, k0, v0, attn.scale, attn.num_queries_per_kv)
    exp1 = _naive_causal_attention(q1, k1, v1, attn.scale, attn.num_queries_per_kv)

    kv_cache = torch.empty(2, 8, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache

    store_kvcache(k0[:6], v0[:6], kv_cache, torch.arange(6))
    store_kvcache(k1[:5], v1[:5], kv_cache, torch.arange(16, 21))

    q = torch.cat([q0[6:7], q1[5:6]], dim=0)
    k = torch.cat([k0[6:7], k1[5:6]], dim=0)
    v = torch.cat([v0[6:7], v1[5:6]], dim=0)

    md = AttentionMetadata(
        slot_mapping=torch.tensor([6, 21]),
        block_tables=torch.tensor([
            [0, 1],
            [4, 5],
        ], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7, 6], dtype=torch.int32),
        context_lens=torch.tensor([6, 5], dtype=torch.int32),
        query_start_loc_cpu=[0, 1, 2],
        seq_lens_cpu=[7, 6],
        context_lens_cpu=[6, 5],
        max_query_len=1,
        max_seq_len=7,
    )

    with set_forward_context(md):
        actual = attn(q, k, v)

    torch.testing.assert_close(actual[0], exp0[6], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1], exp1[5], atol=1e-5, rtol=1e-5)


def test_batched_mixed():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 4
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    kv_cache = torch.empty(2, 6, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    q0 = torch.randn(5, num_heads, head_dim)
    k0 = torch.randn(5, num_kv_heads, head_dim)
    v0 = torch.randn(5, num_kv_heads, head_dim)
    
    k1_ctx = torch.randn(3, num_kv_heads, head_dim)
    v1_ctx = torch.randn(3, num_kv_heads, head_dim)
    store_kvcache(k1_ctx, v1_ctx, kv_cache, torch.arange(8, 11))
    
    q1 = torch.randn(1, num_heads, head_dim)
    k1 = torch.randn(1, num_kv_heads, head_dim)
    v1 = torch.randn(1, num_kv_heads, head_dim)
    
    k2_ctx = torch.randn(8, num_kv_heads, head_dim)
    v2_ctx = torch.randn(8, num_kv_heads, head_dim)
    store_kvcache(k2_ctx, v2_ctx, kv_cache, torch.arange(12, 20))
    
    q2 = torch.randn(1, num_heads, head_dim)
    k2 = torch.randn(1, num_kv_heads, head_dim)
    v2 = torch.randn(1, num_kv_heads, head_dim)
    
    q_batch = torch.cat([q0, q1, q2], dim=0)
    k_batch = torch.cat([k0, k1, k2], dim=0)
    v_batch = torch.cat([v0, v1, v2], dim=0)
    
    md = AttentionMetadata(
        slot_mapping=torch.tensor([0,1,2,3,4, 11, 20]),
        block_tables=torch.tensor([
            [0, 1, -1],
            [2, -1, -1],
            [3, 4, 5]
        ], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5, 6, 7], dtype=torch.int32),
        seq_lens=torch.tensor([5, 4, 9], dtype=torch.int32),
        context_lens=torch.tensor([0, 3, 8], dtype=torch.int32),
        query_start_loc_cpu=[0, 5, 6, 7],
        seq_lens_cpu=[5, 4, 9],
        context_lens_cpu=[0, 3, 8],
        max_query_len=5,
        max_seq_len=9,
    )
    
    with set_forward_context(md):
        actual = attn(q_batch, k_batch, v_batch)
        
    exp0 = _naive_causal_attention(q0, k0, v0, attn.scale, attn.num_queries_per_kv)
    
    exp1 = _naive_causal_attention(
        torch.cat([torch.zeros(3, num_heads, head_dim), q1], dim=0),
        torch.cat([k1_ctx, k1], dim=0),
        torch.cat([v1_ctx, v1], dim=0),
        attn.scale, attn.num_queries_per_kv
    )
    
    exp2 = _naive_causal_attention(
        torch.cat([torch.zeros(8, num_heads, head_dim), q2], dim=0),
        torch.cat([k2_ctx, k2], dim=0),
        torch.cat([v2_ctx, v2], dim=0),
        attn.scale, attn.num_queries_per_kv
    )
    
    torch.testing.assert_close(actual[0:5], exp0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[5:6], exp1[3:4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[6:7], exp2[8:9], atol=1e-5, rtol=1e-5)

def test_non_contiguous():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 2
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    
    q_all = torch.randn(5, num_heads, head_dim)
    k_all = torch.randn(5, num_kv_heads, head_dim)
    v_all = torch.randn(5, num_kv_heads, head_dim)
    
    expected = _naive_causal_attention(q_all, k_all, v_all, attn.scale, attn.num_queries_per_kv)
    
    kv_cache = torch.empty(2, 10, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    slot_mapping = torch.tensor([
        7*2 + 0, 7*2 + 1,
        2*2 + 0, 2*2 + 1,
        5*2 + 0
    ])
    
    md = AttentionMetadata(
        slot_mapping=slot_mapping,
        block_tables=torch.tensor([[7, 2, 5]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        context_lens=torch.tensor([0], dtype=torch.int32),
        query_start_loc_cpu=[0, 5],
        seq_lens_cpu=[5],
        context_lens_cpu=[0],
        max_query_len=5,
        max_seq_len=5,
    )
    
    with set_forward_context(md):
        actual = attn(q_all, k_all, v_all)
        
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_prefill_parity()
    test_prefill_then_decode_parity()
    test_batched_decode()
    test_batched_mixed()
    test_non_contiguous()
    print("All attention tests passed!")

```

test_generation.py

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from babyvllm.llm import LLM
from babyvllm.sequence import SamplingParams

def test_e2e_generation():
    MODEL = "Qwen/Qwen2-0.5B"
    prompts = [
        "The capital of France is",
        "A recipe for chocolate cake:"
    ]
    
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    hf_outputs = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.inference_mode():
            out_ids = hf_model.generate(ids, max_new_tokens=10, do_sample=False)
        out_ids = out_ids[0][ids.shape[1]:]
        hf_outputs.append(tokenizer.decode(out_ids))
        
    my_llm = LLM(MODEL)
    params = SamplingParams(temperature=0.0, max_tokens=10)
    my_outputs = my_llm.generate(prompts, sampling_params=params)
    
    my_outputs_sorted = sorted(my_outputs, key=lambda x: x.seq_id)
    
    for i in range(len(prompts)):
        print(f"--- Prompt: {prompts[i]} ---")
        print(f"HF Output: {hf_outputs[i]}")
        print(f"My Output: {my_outputs_sorted[i].text}")
        assert hf_outputs[i] == my_outputs_sorted[i].text
        
    print("End-to-end greedy generation test passed!")

if __name__ == "__main__":
    test_e2e_generation()
```

test_parity.py

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from babyvllm.config import ModelConfig
from babyvllm.models.qwen2 import Qwen2ForCausalLM
from babyvllm.worker.loader import load_model

def test_parity():
    MODEL = "Qwen/Qwen2-0.5B"
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    
    path = snapshot_download(repo_id=MODEL)
    
    cfg = ModelConfig()
    mine = Qwen2ForCausalLM(cfg)
    load_model(mine, path)
    mine.eval()

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0]
    
    with torch.inference_mode():
        ref = hf(ids[None]).logits[0]
        hidden = mine(ids, torch.arange(len(ids)))
        got = mine.compute_logits(hidden)
        
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
    print("Logit parity test passed!")

if __name__ == "__main__":
    test_parity()
```

test_sampler.py

```python
import torch
import torch.nn.functional as F

def brute_force_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_sum = 0.0
    allowed_indices = []
    
    for i in range(len(sorted_probs)):
        allowed_indices.append(sorted_indices[i].item())
        cum_sum += sorted_probs[i].item()
        if cum_sum > top_p:
            break
            
    out_probs = torch.zeros_like(probs)
    for idx in allowed_indices:
        out_probs[idx] = probs[idx]
        
    return out_probs / out_probs.sum()

def get_vllm_top_p_probs(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = probs.clone()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum()
    return probs

def test_top_p_sampling():
    torch.manual_seed(42)
    for _ in range(100):
        logits = torch.randn(1000)
        probs = F.softmax(logits, dim=-1)
        
        top_p = torch.rand(1).item() * 0.9 + 0.1
        
        expected = brute_force_top_p(probs, top_p)
        actual = get_vllm_top_p_probs(probs, top_p)
        
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        
def test_top_p_edge_cases():
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    
    expected = brute_force_top_p(probs, 0.1)
    actual = get_vllm_top_p_probs(probs, 0.1)
    torch.testing.assert_close(actual, expected)
    
    expected = brute_force_top_p(probs, 1.0)
    actual = get_vllm_top_p_probs(probs, 1.0)
    torch.testing.assert_close(actual, expected)
```

graph_runner.py

```python
from __future__ import annotations

import torch


class CUDAGraphRunner:

    def __init__(self):
        self._graph: torch.cuda.CUDAGraph | None = None

    @property
    def pool(self):
        if self._graph is None:
            raise RuntimeError("Call capture() first")
        return self._graph.pool()

    def capture(self, fn, buf, batch_bucket: int, ctx_bucket: int, pool=None):
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=pool):
            fn(buf, batch_bucket, ctx_bucket)

    def replay(self):
        if self._graph is None:
            raise RuntimeError("Call capture() before replay()")
        self._graph.replay()

```

test_cuda_graphs.py

```python
from babyvllm.worker.model_runner import BATCH_BUCKETS, CTX_BUCKETS, ceil_to_bucket


def test_ceil_to_bucket():
    assert ceil_to_bucket(1, BATCH_BUCKETS) == 1
    assert ceil_to_bucket(3, BATCH_BUCKETS) == 4
    assert ceil_to_bucket(64, BATCH_BUCKETS) == 64
    assert ceil_to_bucket(65, BATCH_BUCKETS) is None
    assert ceil_to_bucket(128, CTX_BUCKETS) == 256
    assert ceil_to_bucket(256, CTX_BUCKETS) == 256
    assert ceil_to_bucket(257, CTX_BUCKETS) == 2048
    assert ceil_to_bucket(2049, CTX_BUCKETS) is None

```
