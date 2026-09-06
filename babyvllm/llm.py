import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from babyvllm.config import ModelConfig, CacheConfig, SchedulerConfig
from babyvllm.engine import LLMEngine
from babyvllm.worker.model_runner import ModelRunner
from babyvllm.worker.loader import load_model
from babyvllm.sequence import SamplingParams

class LLM:
    def __init__(
        self,
        model_name: str,
        cache_config: CacheConfig = None,
        scheduler_config: SchedulerConfig = None
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        try:
            from transformers import GenerationConfig
            gen_cfg = GenerationConfig.from_pretrained(model_name)
            if isinstance(gen_cfg.eos_token_id, list):
                self.eos_token_id = set(gen_cfg.eos_token_id)
            elif isinstance(gen_cfg.eos_token_id, int):
                self.eos_token_id = {gen_cfg.eos_token_id}
            else:
                self.eos_token_id = set()
        except Exception:
            if self.tokenizer.eos_token_id is not None:
                self.eos_token_id = {self.tokenizer.eos_token_id}
            else:
                self.eos_token_id = set()
        
        path = snapshot_download(repo_id=model_name)
        
        import json
        with open(f"{path}/config.json", "r") as f:
            hf_cfg = json.load(f)
            
        self.model_config = ModelConfig(
            vocab_size=hf_cfg.get("vocab_size", 151936),
            hidden_size=hf_cfg.get("hidden_size", 896),
            intermediate_size=hf_cfg.get("intermediate_size", 4864),
            num_hidden_layers=hf_cfg.get("num_hidden_layers", 24),
            num_attention_heads=hf_cfg.get("num_attention_heads", 14),
            num_key_value_heads=hf_cfg.get("num_key_value_heads", 2),
            rms_norm_eps=hf_cfg.get("rms_norm_eps", 1e-6),
            max_position_embeddings=hf_cfg.get("max_position_embeddings", 32768),
            rope_theta=hf_cfg.get("rope_theta", 1000000.0),
            tie_word_embeddings=hf_cfg.get("tie_word_embeddings", True),
        )
        
        sched_cfg = scheduler_config or SchedulerConfig()
        
        self.model_runner = ModelRunner(
            self.model_config,
            cache_config=cache_config,
            scheduler_config=sched_cfg
        )
        load_model(self.model_runner.model, path)
        
        self.engine = LLMEngine(
            model_runner=self.model_runner,
            max_num_batched_tokens=sched_cfg.max_num_batched_tokens,
            max_num_seqs=sched_cfg.max_num_seqs
        )

    def generate(self, prompts: list[str], sampling_params=None):
        import copy
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0)

        for prompt in prompts:
            req_params = copy.copy(sampling_params)
            if not req_params.ignore_eos:
                req_params.stop_tokens = set(req_params.stop_tokens) | self.eos_token_id
            
            token_ids = self.tokenizer(prompt).input_ids
            self.engine.add_request(token_ids, req_params)
            
        outputs = list(self.engine.run().values())
        for out in outputs:
            out.text = self.tokenizer.decode(out.new_token_ids)
            
        return outputs
