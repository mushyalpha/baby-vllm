import torch
from babyvllm.kernels.layernorm import fused_add_rms_norm

def ref(x, res, w, eps):
    h = x if res is None else (x + res)          # bf16 add, like the old path
    hf = h.float()
    y = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)
    return w * y.to(h.dtype), h

torch.manual_seed(0)
H, eps = 3584, 1e-6
w = torch.randn(H, device="cuda", dtype=torch.bfloat16)
for T in [1, 7, 64, 513]:
    x   = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    # with residual
    y_ref, h_ref = ref(x.clone(), res.clone(), w, eps)
    r = res.clone()
    y = fused_add_rms_norm(x.clone(), r, w, eps)
    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(r, h_ref, atol=2e-2, rtol=2e-2)   # residual updated in place

    # residual=None: input must NOT be clobbered
    x0 = x.clone()
    y = fused_add_rms_norm(x0, None, w, eps)
    y_ref, _ = ref(x, None, w, eps)
    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x0, x)                             # catches the aliasing bug
print("PASS: Triton kernel matches PyTorch mathematically and handles memory perfectly.")
