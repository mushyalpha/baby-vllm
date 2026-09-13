import re

with open("babyvllm/layers/attention.py", "r") as f:
    content = f.read()

# Remove the top-level import
content = content.replace("from babyvllm.kernels.paged_attn import paged_decode_attention\n", "")

# Replace the forward logic
old_forward = r"if md\.max_query_len == 1:\n\s+return paged_decode_attention\(q, self\.kv_cache, md\.block_tables, md\.seq_lens, self\.scale\)"
new_forward = """if md.max_query_len == 1 and q.is_cuda and self.kv_cache.shape[2] >= 16:
            from babyvllm.kernels.paged_attn import paged_decode_attention
            return paged_decode_attention(q, self.kv_cache, md.block_tables, md.seq_lens, self.scale)
        if md.max_query_len == 1:
            return _static_decode_attention(
                q, self.kv_cache, md, self.scale,
                self.num_heads, self.num_kv_heads, self.head_dim,
                self.num_queries_per_kv,
            )"""
            
content = re.sub(old_forward, new_forward, content)

with open("babyvllm/layers/attention.py", "w") as f:
    f.write(content)
