with open("babyvllm/layers/sampler.py", "r") as f:
    content = f.read()

new_content = """import torch
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
        seqs = scheduler_output.scheduled_sequences
        if not seqs:
            return {}

        B = logits.size(0)
        tokens = torch.empty(B, dtype=torch.int64, device=logits.device)
        logits = logits.float()
        
        greedy_indices = []
        stoch_indices = []
        for i, seq in enumerate(seqs):
            p = seq.sampling_params
            if p is None or p.temperature == 0.0:
                greedy_indices.append(i)
            else:
                stoch_indices.append(i)
                
        if greedy_indices:
            idx = torch.tensor(greedy_indices, device=logits.device)
            tokens[idx] = torch.argmax(logits[idx], dim=-1)
            
        if stoch_indices:
            for i in stoch_indices:
                p = seqs[i].sampling_params
                logit = logits[i]
                probs = F.softmax(logit / p.temperature, dim=-1)
                if p.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > p.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum()
                tokens[i] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # ONE SINGLE CPU SYNC
        tokens_cpu = tokens.cpu().tolist()
        
        sampled_tokens = {}
        for i, seq in enumerate(seqs):
            sampled_tokens[seq.seq_id] = tokens_cpu[i]
            
        return sampled_tokens
"""

with open("babyvllm/layers/sampler.py", "w") as f:
    f.write(new_content)
