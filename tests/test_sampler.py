import torch
import torch.nn.functional as F

def brute_force_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_sum = 0.0
    allowed_indices = []
    
    for i in range(len(sorted_probs)):
        allowed_indices.append(sorted_indices[i].item())
        cum_sum += sorted_probs[i].item()
        if cum_sum > top_p:
            break
            
    out_probs = torch.zeros_like(probs)
    for idx in allowed_indices:
        out_probs[idx] = probs[idx]
        
    return out_probs / out_probs.sum()

def get_vllm_top_p_probs(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = probs.clone()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum()
    return probs

def test_top_p_sampling():
    torch.manual_seed(42)
    for _ in range(100):
        logits = torch.randn(1000)
        probs = F.softmax(logits, dim=-1)
        
        top_p = torch.rand(1).item() * 0.9 + 0.1
        
        expected = brute_force_top_p(probs, top_p)
        actual = get_vllm_top_p_probs(probs, top_p)
        
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        
def test_top_p_edge_cases():
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    
    expected = brute_force_top_p(probs, 0.1)
    actual = get_vllm_top_p_probs(probs, 0.1)
    torch.testing.assert_close(actual, expected)
    
    expected = brute_force_top_p(probs, 1.0)
    actual = get_vllm_top_p_probs(probs, 1.0)
    torch.testing.assert_close(actual, expected)
