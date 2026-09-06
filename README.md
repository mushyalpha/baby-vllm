# Baby-vLLM

Interactive diagram of a vLLM V1 engine step: continuous batching, mixed prefill/decode, KV pressure, preemption, and block reuse.

```bash
npm start
```

## Architecture Notes

### Paged Attention KV Cache Allocation

In `PagedAttention`, the current implementation performs an advanced-index copy of the whole sequence KV into a newly allocated tensor:

```python
k_i = kv_cache[0][blocks].view(-1, num_kv_heads, head_dim)[:seq_len]
v_i = kv_cache[1][blocks].view(-1, num_kv_heads, head_dim)[:seq_len]
```

While functionally correct for a v1 implementation to test the engine, **this allocation-heavy pattern is the primary reason FlashAttention's paged kernel exists in production vLLM.** The custom CUDA kernel in vLLM reads directly from the disjoint block tables during attention computation without needing to materialize a contiguous tensor in HBM.
