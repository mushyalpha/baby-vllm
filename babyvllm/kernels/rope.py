import torch
import triton
import triton.language as tl

@triton.jit
def _rope_kv_cache_kernel(
    q_ptr, k_ptr, v_ptr,
    cos_ptr, sin_ptr,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_cost, stride_cosd,
    stride_kct, stride_kch, stride_kcd,
    stride_vct, stride_vch, stride_vcd,
    num_kv_heads: tl.constexpr, 
    head_dim: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # We process half of the head_dim at a time for (x1, x2) pairs
    HALF_D: tl.constexpr = head_dim // 2
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HALF_D
    
    # Pointers for Q
    q_base = q_ptr + token_idx * stride_qt + head_idx * stride_qh
    q1_ptrs = q_base + offsets * stride_qd
    q2_ptrs = q_base + (HALF_D + offsets) * stride_qd
    
    # Pointers for cos/sin
    cos_base = cos_ptr + token_idx * stride_cost
    sin_base = sin_ptr + token_idx * stride_cost
    cos1_ptrs = cos_base + offsets * stride_cosd
    cos2_ptrs = cos_base + (HALF_D + offsets) * stride_cosd
    sin1_ptrs = sin_base + offsets * stride_cosd
    sin2_ptrs = sin_base + (HALF_D + offsets) * stride_cosd
    
    # Load Q, cos, sin
    q1 = tl.load(q1_ptrs, mask=mask)
    q2 = tl.load(q2_ptrs, mask=mask)
    cos1 = tl.load(cos1_ptrs, mask=mask)
    cos2 = tl.load(cos2_ptrs, mask=mask)
    sin1 = tl.load(sin1_ptrs, mask=mask)
    sin2 = tl.load(sin2_ptrs, mask=mask)
    
    # Compute RoPE for Q
    # out1 = x1 * cos - x2 * sin
    # out2 = x2 * cos + x1 * sin
    q1_out = q1 * cos1 - q2 * sin1
    q2_out = q2 * cos2 + q1 * sin2
    
    # Store Q back in-place
    tl.store(q1_ptrs, q1_out, mask=mask)
    tl.store(q2_ptrs, q2_out, mask=mask)
    
    # Process K and V only for valid KV heads
    if head_idx < num_kv_heads:
        k_base = k_ptr + token_idx * stride_kt + head_idx * stride_kh
        k1_ptrs = k_base + offsets * stride_kd
        k2_ptrs = k_base + (HALF_D + offsets) * stride_kd
        
        v_base = v_ptr + token_idx * stride_vt + head_idx * stride_vh
        v_ptrs = v_base + tl.arange(0, BLOCK_SIZE * 2)
        v_mask = tl.arange(0, BLOCK_SIZE * 2) < head_dim
        
        # Load K and V
        k1 = tl.load(k1_ptrs, mask=mask)
        k2 = tl.load(k2_ptrs, mask=mask)
        v_full = tl.load(v_ptrs, mask=v_mask)
        
        # Compute RoPE for K
        k1_out = k1 * cos1 - k2 * sin1
        k2_out = k2 * cos2 + k1 * sin2
        
        # Store K back in-place
        tl.store(k1_ptrs, k1_out, mask=mask)
        tl.store(k2_ptrs, k2_out, mask=mask)
        
        if HAS_CACHE:
            slot_idx = tl.load(slot_mapping_ptr + token_idx)
            
            kc_base = k_cache_ptr + slot_idx * stride_kct + head_idx * stride_kch
            kc1_ptrs = kc_base + offsets * stride_kcd
            kc2_ptrs = kc_base + (HALF_D + offsets) * stride_kcd
            
            vc_base = v_cache_ptr + slot_idx * stride_vct + head_idx * stride_vch
            vc_ptrs = vc_base + tl.arange(0, BLOCK_SIZE * 2)
            
            # Write to KV Cache
            tl.store(kc1_ptrs, k1_out, mask=mask)
            tl.store(kc2_ptrs, k2_out, mask=mask)
            tl.store(vc_ptrs, v_full, mask=v_mask)


def fused_rope_and_cache(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    kv_cache: torch.Tensor = None, slot_mapping: torch.Tensor = None
):
    """
    Applies RoPE to Q and K in-place, and optionally writes K and V directly to the paged KV cache.
    q: [num_tokens, num_heads, head_dim]
    k: [num_tokens, num_kv_heads, head_dim]
    v: [num_tokens, num_kv_heads, head_dim]
    """
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    
    # We process half a head at a time, so BLOCK_SIZE must cover head_dim // 2
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
    
    has_cache = (kv_cache is not None and slot_mapping is not None)
    if has_cache:
        # kv_cache shape is [2, num_blocks, block_size, num_kv_heads, head_dim]
        # flat shape is [num_blocks * block_size, num_kv_heads, head_dim]
        k_cache = kv_cache[0].view(-1, num_kv_heads, head_dim)
        v_cache = kv_cache[1].view(-1, num_kv_heads, head_dim)
    else:
        k_cache = k # dummy
        v_cache = v # dummy
        
    grid = (num_tokens, num_heads)
    
    _rope_kv_cache_kernel[grid](
        q, k, v,
        cos, sin,
        k_cache if has_cache else None,
        v_cache if has_cache else None,
        slot_mapping if has_cache else None,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        cos.stride(0), cos.stride(1),
        k_cache.stride(0) if has_cache else 0,
        k_cache.stride(1) if has_cache else 0,
        k_cache.stride(2) if has_cache else 0,
        v_cache.stride(0) if has_cache else 0,
        v_cache.stride(1) if has_cache else 0,
        v_cache.stride(2) if has_cache else 0,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        HAS_CACHE=has_cache,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return q, k
