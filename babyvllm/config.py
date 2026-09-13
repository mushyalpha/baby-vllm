import json
import os
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

# Published numbers: this model, bf16, H100. 0.5B is tests/kernel iteration only.
PUBLISHED_MODEL = "Qwen/Qwen2.5-7B"
DEV_MODEL = "Qwen/Qwen2.5-0.5B"


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope_theta: float
    tie_word_embeddings: bool
    block_size: int = 16

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_bytes_per_token(self) -> int:
        # K+V, all layers, bf16. Published numbers are always bf16.
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * 2

    @classmethod
    def from_hf(cls, path: str, block_size: int = 16) -> "ModelConfig":
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        assert c["architectures"] == ["Qwen2ForCausalLM"], c["architectures"]
        hidden = c["hidden_size"]
        n_heads = c["num_attention_heads"]
        assert hidden % n_heads == 0, (hidden, n_heads)
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=hidden,
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=c["num_key_value_heads"],
            rms_norm_eps=c["rms_norm_eps"],
            max_position_embeddings=c["max_position_embeddings"],
            rope_theta=c["rope_theta"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            block_size=block_size,
        )


def load_model_config(model: str | None = None, block_size: int = 16) -> ModelConfig:
    """Read architecture from checkpoint config.json. Never from hardcoded dims."""
    model = model or os.environ.get("BABYVLLM_MODEL", PUBLISHED_MODEL)
    if os.path.isdir(model):
        return ModelConfig.from_hf(model, block_size=block_size)
    cfg_file = hf_hub_download(repo_id=model, filename="config.json")
    return ModelConfig.from_hf(os.path.dirname(cfg_file), block_size=block_size)


@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int = 100
    num_cpu_blocks: int = 100


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
