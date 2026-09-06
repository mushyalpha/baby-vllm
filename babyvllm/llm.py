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
        
        self.model_config = ModelConfig()
        
        path = snapshot_download(repo_id=model_name)
        
        self.model_runner = ModelRunner(self.model_config)
        load_model(self.model_runner.model, path)
        
        sched_cfg = scheduler_config or SchedulerConfig()
        
        self.engine = LLMEngine(
            model_runner=self.model_runner,
            max_num_batched_tokens=sched_cfg.max_num_batched_tokens,
            max_num_seqs=sched_cfg.max_num_seqs
        )

    def generate(self, prompts: list[str], sampling_params=None):
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0)

        for prompt in prompts:
            token_ids = self.tokenizer(prompt).input_ids
            self.engine.add_request(token_ids, sampling_params)
            
        final_outputs = {}
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for out in step_outputs:
                if out.seq_id not in final_outputs:
                    from babyvllm.engine import RequestOutput
                    final_outputs[out.seq_id] = RequestOutput(
                        seq_id=out.seq_id,
                        new_token_ids=[],
                        finish_reason=None
                    )
                final_outputs[out.seq_id].new_token_ids.extend(out.new_token_ids)
                if out.finished:
                    final_outputs[out.seq_id].finish_reason = out.finish_reason
        
        outputs = list(final_outputs.values())
        for out in outputs:
            out.text = self.tokenizer.decode(out.new_token_ids)
            
        return outputs
