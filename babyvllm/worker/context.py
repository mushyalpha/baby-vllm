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
