import torch
import triton
import triton.language as tl

@triton.jit
def _silu_mul_kernel(
    gate_up_ptr, output_ptr,
    total_elements, D,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate row and col in the flattened (M, D) output space
    row = offsets // D
    col = offsets % D
    
    # Map back to the (M, 2D) input space
    # gate is at [row, col], up is at [row, D + col]
    gate_idx = row * (2 * D) + col
    up_idx = gate_idx + D
    
    # Load gate and up
    gate = tl.load(gate_up_ptr + gate_idx, mask=mask, other=0.0)
    up = tl.load(gate_up_ptr + up_idx, mask=mask, other=0.0)
    
    # SiLU: x * sigmoid(x) -> done in fp32 for precision
    gate_fp32 = gate.to(tl.float32)
    # Triton math.exp requires fp32
    sigmoid = 1.0 / (1.0 + tl.math.exp(-gate_fp32))
    silu = gate_fp32 * sigmoid
    
    # Multiply with up
    up_fp32 = up.to(tl.float32)
    out = silu * up_fp32
    
    # Store result
    tl.store(output_ptr + offsets, out.to(gate.dtype), mask=mask)

def fused_silu_mul(gate_up: torch.Tensor):
    """
    Fuses F.silu(gate) * up where gate_up is a concatenated tensor (..., 2 * D)
    """
    assert gate_up.is_contiguous(), "gate_up tensor must be contiguous"
    
    original_shape = gate_up.shape[:-1]
    D = gate_up.shape[-1] // 2
    
    M = gate_up.numel() // (2 * D)
    total_elements = M * D
    
    output = torch.empty((M, D), device=gate_up.device, dtype=gate_up.dtype)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    _silu_mul_kernel[grid](
        gate_up, output,
        total_elements, D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.view(*original_shape, D)
