import itertools
from enum import Enum, auto
from dataclasses import dataclass, field

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 256
    stop_tokens: set[int] = field(default_factory=set)


class SequenceState(Enum):
    WAITING = auto()
    RUNNING = auto()
    DONE = auto()

class SequenceFinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"

class Sequence:
    _counter = itertools.count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        if not token_ids:
            raise ValueError("Prompt cannot be empty")
        self.seq_id = next(Sequence._counter)
        self.token_ids = list(token_ids)
        self.sampling_params = sampling_params or SamplingParams()
        self.status = SequenceState.WAITING
        self.finish_reason = None
        self._prompt_len = len(token_ids)
        self.num_computed_tokens: int = 0


    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, item):
        return self.token_ids[item]

    def __repr__(self) -> str:
        return (f"Sequence(id={self.seq_id}, {self.status.name}, "
                f"prompt={self._prompt_len}, total={len(self)}, "
                f"computed={self.num_computed_tokens})")

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self._prompt_len:]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def advance_computed(self, n: int) -> None:
        self.num_computed_tokens += n
        if self.num_computed_tokens > len(self.token_ids):
            raise ValueError(f"Computed tokens ({self.num_computed_tokens}) exceeds sequence length ({len(self.token_ids)})")

    @property
    def prompt_len(self) -> int:
        return self._prompt_len

    @property
    def generated_token_len(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def num_tokens_to_compute(self) -> int:
        return len(self.token_ids) - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceState.DONE

    @property
    def last_token_id(self):
        return self.token_ids[-1]

    def finish(self, reason: SequenceFinishReason) -> None:
        self.status = SequenceState.DONE
        self.finish_reason = reason

    def reset_for_recompute(self) -> None:
        self.status = SequenceState.WAITING
        self.num_computed_tokens = 0
