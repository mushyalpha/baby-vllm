import torch
import torch.nn.functional as F
from unittest.mock import patch
from dataclasses import dataclass

from babyvllm.layers.sampler import Sampler
from babyvllm.sequence import Sequence, SamplingParams

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

@dataclass
class MockSchedulerOutput:
    scheduled_sequences: list

def test_top_p_sampling():
    torch.manual_seed(42)
    sampler = Sampler(vocab_size=1000)
    
    for _ in range(100):
        logits = torch.randn(1, 1000)
        probs = F.softmax(logits, dim=-1)
        
        top_p = torch.rand(1).item() * 0.9 + 0.1
        expected = brute_force_top_p(probs[0], top_p)
        
        seq = Sequence([1], sampling_params=SamplingParams(top_p=top_p, temperature=1.0))
        scheduler_out = MockSchedulerOutput([seq])
        
        with patch("torch.multinomial") as mock_multinomial:
            mock_multinomial.return_value = torch.tensor([[0]])
            sampler(logits, None, scheduler_out)
            actual_probs = mock_multinomial.call_args[0][0]
            
        torch.testing.assert_close(actual_probs, expected, rtol=1e-5, atol=1e-5)

def test_top_p_edge_cases():
    sampler = Sampler(vocab_size=4)
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    logits = torch.log(probs)
    
    # top_p = 0.1
    expected = brute_force_top_p(probs[0], 0.1)
    seq = Sequence([1], sampling_params=SamplingParams(top_p=0.1, temperature=1.0))
    scheduler_out = MockSchedulerOutput([seq])
    with patch("torch.multinomial") as mock_multinomial:
        mock_multinomial.return_value = torch.tensor([[0]])
        sampler(logits, None, scheduler_out)
        actual = mock_multinomial.call_args[0][0]
    torch.testing.assert_close(actual, expected)
    
    # top_p = 1.0
    expected = brute_force_top_p(probs[0], 1.0)
    seq = Sequence([1], sampling_params=SamplingParams(top_p=1.0, temperature=1.0))
    scheduler_out = MockSchedulerOutput([seq])
    with patch("torch.multinomial") as mock_multinomial:
        mock_multinomial.return_value = torch.tensor([[0]])
        sampler(logits, None, scheduler_out)
        actual = mock_multinomial.call_args[0][0]
    torch.testing.assert_close(actual, expected)
