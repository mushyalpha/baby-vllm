from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    block_size: int = 16
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int = 100
    num_cpu_blocks: int = 100

@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048

