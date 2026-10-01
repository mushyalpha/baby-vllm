import torch
import triton
import triton.language as tl

@triton.jit
def _fused_add_rms_norm_kernel(
    input_ptr, residual_ptr, weight_ptr, output_ptr,
    stride_input_m, stride_residual_m, stride_output_m,
    N, eps,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    input_ptr += row_idx * stride_input_m
    output_ptr += row_idx * stride_output_m
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input (e.g. output from attention or MLP)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if HAS_RESIDUAL:
        residual_ptr += row_idx * stride_residual_m
        res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        # Add residual
        x = x + res
        # In-place write back to residual
        tl.store(residual_ptr + offsets, x, mask=mask)
        
    # RMSNorm computation (float32 for numerical stability)
    x_fp32 = x.to(tl.float32)
    variance = tl.sum(x_fp32 * x_fp32, axis=0) / N
    rsqrt = tl.math.rsqrt(variance + eps)
    
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    out = (x_fp32 * rsqrt).to(x.dtype) * w
    
    tl.store(output_ptr + offsets, out, mask=mask)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float):
    original_shape = input.shape
    
    # CRITICAL: Do not change .view() to .reshape()!
    # If the tensor is non-contiguous, .view() will loudly fail. 
    # .reshape() would silently copy, causing the in-place residual update to write to a temporary tensor!
    input_2d = input.view(-1, input.shape[-1])
    assert input_2d.stride(-1) == 1, "Last dimension must be contiguous"
    
    if residual is not None:
        residual_2d = residual.view(-1, residual.shape[-1])
        assert residual_2d.stride(-1) == 1, "Residual last dimension must be contiguous"
    else:
        residual_2d = None
        
    M, N = input_2d.shape
    output_2d = torch.empty_like(input_2d)
    
    MAX_FUSED_SIZE = 65536
    block_size = triton.next_power_of_2(N)
    if block_size > MAX_FUSED_SIZE:
        raise RuntimeError(f"Hidden size {N} too large for fused RMSNorm")
        
    num_warps = 8 if block_size >= 4096 else 4
    has_residual = residual_2d is not None
    
    _fused_add_rms_norm_kernel[(M,)](
        input_2d,
        residual_2d,
        weight,
        output_2d,
        input_2d.stride(0),
        residual_2d.stride(0) if has_residual else 0,
        output_2d.stride(0),
        N,
        eps,
        HAS_RESIDUAL=has_residual,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output_2d.view(original_shape)
