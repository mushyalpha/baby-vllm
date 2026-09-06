import argparse
import os
import sys
import time

# python3 benchmarks/benchmark_throughput.py puts `benchmarks/` on sys.path,
# not the repo root — so `import babyvllm` fails without this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from babyvllm.engine import RequestOutput
from babyvllm.llm import LLM
from babyvllm.sequence import SamplingParams
from babyvllm.worker.model_runner import pick_device, pick_dtype


def parse_args():
    parser = argparse.ArgumentParser(
        description="Throughput: HuggingFace sequential generate vs Baby-vLLM"
    )
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--prompt-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else pick_device()
    dtype = pick_dtype(device)
    on_cuda = device.type == "cuda"

    num_requests = args.num_requests if args.num_requests is not None else (100 if on_cuda else 4)
    prompt_len = args.prompt_len if args.prompt_len is not None else (64 if on_cuda else 32)
    max_tokens = args.max_tokens if args.max_tokens is not None else (64 if on_cuda else 16)

    print(f"Device: {device}  dtype: {dtype}", flush=True)
    print(
        f"Workload: {num_requests} requests, prompt_len={prompt_len}, max_tokens={max_tokens}",
        flush=True,
    )
    if device.type == "cpu":
        print(
            "Running on CPU — using a small workload. "
            "100 requests × 64 tokens can take an hour+ and used to print nothing until the end.",
            flush=True,
        )
    elif device.type == "mps":
        print("Using Apple GPU (MPS).", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dummy_input = "The quick brown fox jumps over the lazy dog. " * (prompt_len // 8)
    prompts = [dummy_input] * num_requests

    hf_throughput = None
    if not args.skip_hf:
        print("\n[1/2] HuggingFace Transformers (one request at a time)...", flush=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype
        ).to(device).eval()

        start_time = time.time()
        total_hf_tokens = 0
        with torch.inference_mode():
            for i, prompt in enumerate(prompts, 1):
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                out = hf_model.generate(
                    ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                total_hf_tokens += out.shape[1] - ids.shape[1]
                print(
                    f"  HF {i}/{num_requests}  {time.time() - start_time:.1f}s",
                    flush=True,
                )

        hf_time = time.time() - start_time
        hf_throughput = total_hf_tokens / hf_time
        print(f"-> HF throughput: {hf_throughput:.2f} tok/s  ({hf_time:.1f}s)", flush=True)

        del hf_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    else:
        print("\n[1/2] Skipping HuggingFace baseline (--skip-hf)", flush=True)

    print("\n[2/2] Baby-vLLM (continuous batching)...", flush=True)
    llm = LLM(args.model, device=device, verbose=True)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    for prompt in prompts:
        llm.engine.add_request(llm.tokenizer(prompt).input_ids, params)

    print("Running generate loop (first step is prefill)...", flush=True)
    start_time = time.time()
    final_outputs: dict[int, RequestOutput] = {}
    steps = 0
    while llm.engine.has_unfinished_requests():
        for out in llm.engine.step():
            if out.seq_id not in final_outputs:
                final_outputs[out.seq_id] = RequestOutput(
                    seq_id=out.seq_id,
                    new_token_ids=[],
                    finish_reason=None,
                )
            final_outputs[out.seq_id].new_token_ids.extend(out.new_token_ids)
            if out.finished:
                final_outputs[out.seq_id].finish_reason = out.finish_reason
        steps += 1
        in_flight = len(llm.engine.scheduler.waiting) + len(llm.engine.scheduler.running)
        print(
            f"  step {steps}  in-flight={in_flight}  {time.time() - start_time:.1f}s",
            flush=True,
        )

    vllm_time = time.time() - start_time
    total_vllm_tokens = sum(len(o.new_token_ids) for o in final_outputs.values())
    vllm_throughput = total_vllm_tokens / vllm_time if vllm_time > 0 else 0.0
    print(
        f"-> Baby-vLLM throughput: {vllm_throughput:.2f} tok/s  "
        f"({total_vllm_tokens} tokens, {vllm_time:.1f}s, {steps} steps)",
        flush=True,
    )

    print("\n=========================================")
    if hf_throughput is not None:
        print(f"HF Transformers:   {hf_throughput:.2f} tok/s")
        print(f"Baby-vLLM:         {vllm_throughput:.2f} tok/s")
        print(f"Speedup:           {vllm_throughput / hf_throughput:.2f}x")
    else:
        print(f"Baby-vLLM:         {vllm_throughput:.2f} tok/s")
    print("=========================================")


if __name__ == "__main__":
    main()
