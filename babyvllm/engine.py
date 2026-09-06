from dataclasses import dataclass

from babyvllm.sequence import (
    Sequence,
    SamplingParams,
    SequenceFinishReason,
)
from babyvllm.scheduler import Scheduler, SchedulerOutput
from babyvllm.kv_cache_manager import KVCacheManager


@dataclass
class RequestOutput:
    seq_id: int
    new_token_ids: list[int]
    finish_reason: SequenceFinishReason | None = None
    text: str = ""

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class LLMEngine:
    def __init__(
        self,
        model_runner,
        max_num_batched_tokens: int = 8,
        max_num_seqs: int = 4,
    ):
        self.model_runner = model_runner

        self.kv_cache_manager = KVCacheManager(
            num_blocks=model_runner.determine_num_blocks(),
            block_size=model_runner.block_size,
        )

        self.scheduler = Scheduler(
            kv_cache_manager=self.kv_cache_manager,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )

        self.sequences: dict[int, Sequence] = {}

        self.block_size = model_runner.block_size

    def add_request(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> int:

        seq = Sequence(
            token_ids=token_ids,
            sampling_params=sampling_params,
        )

        self.sequences[seq.seq_id] = seq
        self.scheduler.add_seq(seq)

        return seq.seq_id

    def abort_request(self, seq_id: int) -> None:
        seq = self.sequences.get(seq_id)
        if seq is None:
            return

        if not seq.is_finished:
            self.scheduler.abort_seq(seq)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> list[RequestOutput]:
        if not self.has_unfinished_requests() and not self.scheduler.finished_seqs:
            return []

        scheduler_output = None
        sampled_tokens = {}
        if self.has_unfinished_requests():
            scheduler_output = self.scheduler.schedule()

            made_progress = (
                bool(scheduler_output.scheduled_sequences) or 
                bool(self.scheduler.finished_seqs) or 
                bool(scheduler_output.preempted_seqs)
            )

            if not made_progress:
                raise RuntimeError(
                    "Engine stalled: unfinished requests exist, "
                    "but scheduler scheduled no work, preempted nothing, and no sequences finished."
                )

            if scheduler_output.scheduled_sequences:
                sampled_tokens = self.model_runner.execute_model(
                    scheduler_output
                )

                if not isinstance(sampled_tokens, dict):
                    raise RuntimeError(
                        "ModelRunner returned incorrect type"
                    )

                self.scheduler.update_from_output(
                    scheduler_output,
                    sampled_tokens,
                )

        outputs = []
        handled_seq_ids = set()
        
        if scheduler_output and scheduler_output.scheduled_sequences:
            for seq in scheduler_output.scheduled_sequences:
                if seq.seq_id in sampled_tokens:
                    token = sampled_tokens[seq.seq_id]
                    new_ids = [] if token in seq.sampling_params.stop_tokens else [token]
                    outputs.append(
                        RequestOutput(
                            seq_id=seq.seq_id,
                            new_token_ids=new_ids,
                            finish_reason=seq.finish_reason,
                        )
                    )
                    handled_seq_ids.add(seq.seq_id)

        for seq in self.scheduler.pop_finished():
            if seq.seq_id not in handled_seq_ids:
                outputs.append(
                    RequestOutput(
                        seq_id=seq.seq_id,
                        new_token_ids=[],
                        finish_reason=seq.finish_reason,
                    )
                )
            self.sequences.pop(seq.seq_id, None)

        return outputs

    def run(self, max_steps: int = 1000) -> dict[int, RequestOutput]:
        steps = 0
        final_outputs: dict[int, RequestOutput] = {}
        
        while self.has_unfinished_requests():
            step_outputs = self.step()
            for out in step_outputs:
                if out.seq_id not in final_outputs:
                    final_outputs[out.seq_id] = RequestOutput(
                        seq_id=out.seq_id,
                        new_token_ids=[],
                        finish_reason=None
                    )
                final_outputs[out.seq_id].new_token_ids.extend(out.new_token_ids)
                if out.finished:
                    final_outputs[out.seq_id].finish_reason = out.finish_reason
            
            steps += 1
            if steps >= max_steps:
                raise RuntimeError(f"Engine.run() hit max_steps ({max_steps}). Possible livelock.")

        return final_outputs

    def reset(self) -> None:
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.scheduler.finished_seqs.clear()
        self.kv_cache_manager.reset()
        self.sequences.clear()
