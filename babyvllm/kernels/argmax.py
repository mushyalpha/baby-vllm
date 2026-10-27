import torch
import triton
import triton.language as tl

# Greedy sampling over a (batch, vocab) logits matrix. torch.argmax gives each
# row to one reduction, so a 64 x 152_064 step keeps few SMs busy. This splits
# every row into BLOCK-wide chunks, takes each chunk's max and index, then
# reduces the chunk winners per row.
#
# Ties go to the lowest index in both stages, which is what torch.argmax
# returns, so tokens are identical to the torch path.


@triton.jit
def _argmax_chunks_kernel(
    x_ptr, part_val_ptr, part_idx_ptr,
    n_cols, stride_row, n_chunks,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride_row + offs, mask=offs < n_cols, other=-float("inf"))
    val, idx = tl.max(x.to(tl.float32), axis=0, return_indices=True, return_indices_tie_break_left=True)
    tl.store(part_val_ptr + row * n_chunks + chunk, val)
    tl.store(part_idx_ptr + row * n_chunks + chunk, chunk * BLOCK + idx)


@triton.jit
def _argmax_rows_kernel(
    part_val_ptr, part_idx_ptr, out_ptr,
    n_chunks,
    CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, CHUNKS)
    vals = tl.load(part_val_ptr + row * n_chunks + offs, mask=offs < n_chunks, other=-float("inf"))
    _, best = tl.max(vals, axis=0, return_indices=True, return_indices_tie_break_left=True)
    tok = tl.load(part_idx_ptr + row * n_chunks + best)
    tl.store(out_ptr + row, tok.to(tl.int64))


def greedy_argmax(logits: torch.Tensor, out: torch.Tensor, block: int = 4096) -> torch.Tensor:
    """out[i] = argmax(logits[i]). logits is (rows, vocab) with unit column stride."""
    assert logits.dim() == 2 and logits.stride(1) == 1, "logits must be (rows, vocab), contiguous rows"
    rows, n_cols = logits.shape
    n_chunks = triton.cdiv(n_cols, block)
    part_val = torch.empty(rows, n_chunks, device=logits.device, dtype=torch.float32)
    part_idx = torch.empty(rows, n_chunks, device=logits.device, dtype=torch.int32)
    _argmax_chunks_kernel[(rows, n_chunks)](
        logits, part_val, part_idx, n_cols, logits.stride(0), n_chunks, BLOCK=block,
    )
    _argmax_rows_kernel[(rows,)](
        part_val, part_idx, out, n_chunks, CHUNKS=triton.next_power_of_2(n_chunks),
    )
    return out
