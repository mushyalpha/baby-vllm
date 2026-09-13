import re

with open("babyvllm/layers/attention.py", "r") as f:
    code = f.read()

# Replace _static_decode_attention call with paged_decode_attention
if "from babyvllm.kernels.paged_attn import paged_decode_attention" not in code:
    code = "from babyvllm.kernels.paged_attn import paged_decode_attention\n" + code

code = re.sub(
    r"if md\.max_query_len == 1:.*return _static_decode_attention\([^)]+\)",
    r"if md.max_query_len == 1:\n            return paged_decode_attention(q, self.kv_cache, md.block_tables, md.seq_lens, self.scale)",
    code,
    flags=re.DOTALL
)

with open("babyvllm/layers/attention.py", "w") as f:
    f.write(code)
