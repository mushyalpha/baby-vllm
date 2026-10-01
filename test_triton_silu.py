import torch
import torch.nn.functional as F
from babyvllm.kernels.activation import fused_silu_mul

def ref_silu_mul(gate_up):
    D = gate_up.shape[-1] // 2
    gate, up = gate_up.split(D, dim=-1)
    return F.silu(gate) * up

torch.manual_seed(0)
# intermediate_size for 7B is 18944, so gate_up is 37888
D = 18944
for T in [1, 7, 64, 513]:
    gate_up = torch.randn(T, 2 * D, device="cuda", dtype=torch.bfloat16)
    
    # Reference
    y_ref = ref_silu_mul(gate_up)
    
    # Triton
    y_triton = fused_silu_mul(gate_up)
    
    # FP32 math vs FP32 Triton math might have slight ULP differences
    torch.testing.assert_close(y_triton, y_ref, atol=2e-2, rtol=2e-2)
    
print("PASS: Triton SiLU+Mul kernel matches PyTorch mathematically!")
