import re

with open("babyvllm/kernels/paged_attn.py", "r") as f:
    content = f.read()

# Fix mask
content = content.replace("mask = (token[None, :] < seq_len) & q_mask[:, None]", "mask = token[None, :] < seq_len")

# Fix zeros_like
content = content.replace("out = torch.empty_like(q)", "out = torch.zeros_like(q)")

with open("babyvllm/kernels/paged_attn.py", "w") as f:
    f.write(content)
