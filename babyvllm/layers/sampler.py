import torch
import torch.nn as nn
import torch.nn.functional as F

from babyvllm.worker.context import AttentionMetadata

class Sampler(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        scheduler_output,
    ) -> dict[int, int]:
        sampled_tokens = {}
        for i, seq in enumerate(scheduler_output.scheduled_sequences):
            params = seq.sampling_params
            logit = logits[i].float()
            
            if params is None or params.temperature == 0.0:
                token = torch.argmax(logit).item()
            else:
                probs = F.softmax(logit / params.temperature, dim=-1)
                if params.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > params.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum()
                token = torch.multinomial(probs, num_samples=1).item()
            sampled_tokens[seq.seq_id] = token
            
        return sampled_tokens
