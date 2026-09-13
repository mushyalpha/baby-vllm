import re

with open("babyvllm/worker/model_runner.py", "r") as f:
    content = f.read()

old_func = r"def compute_pool_reserve.*?return total_gather \+ slack"
new_func = """def compute_pool_reserve(
    max_batch_bucket: int,
    max_ctx_bucket: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    bytes_per_element: int = 2,
) -> int:
    slack = 512 * 1024 * 1024  # 512 MB slack for staging buffers and miscellaneous allocations
    return slack"""

content = re.sub(old_func, new_func, content, flags=re.DOTALL)

with open("babyvllm/worker/model_runner.py", "w") as f:
    f.write(content)
