import torch
import torch.nn as nn

try:
    from babyvllm.kernels.layernorm import fused_add_rms_norm
    HAS_TRITON = True
except ImportError as e:
    HAS_TRITON = False
    print(f"[babyvllm] WARNING: Triton RMSNorm unavailable: {e}")

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if x.is_cuda and HAS_TRITON:
            if residual is not None:
                out = fused_add_rms_norm(x, residual, self.weight, self.eps)
                return out, residual
            else:
                residual = x
                out = fused_add_rms_norm(x, None, self.weight, self.eps)
                return out, residual

        # CPU / eager fallback
        if residual is not None:
            residual.add_(x)
            x = residual
        else:
            residual = x

        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.weight * x.to(dt)
        return out, residual
