#!/usr/bin/env python3
"""Run Qwen2.5-7B through baby-vLLM and write down what the engine actually did.

This script does not invent token ids, block ids, or addresses. It loads the
model on the CUDA device it finds, steps the real scheduler, and records the
block table, free list, slot mapping, sampled token ids, and the layer-0 KV
pointers those slots map to.

Usage (from the repo root, on the GPU machine):

    python scripts/trace_batch_on_gpu.py

Writes trace_out/batch_trace.md and trace_out/batch_trace.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAIR = [
    {
        "name": "visas",
        "prompt": "Hey, can you just tell me about the new visas laws So, for example",
        "max_tokens": 8,
    },
    {
        "name": "uber",
        "prompt": "Typically how much does an uber drive earn in a big uk city like Glasgow per day",
        "max_tokens": 36,
    },
]

BATCH = [
    {
        "name": "uber",
        "prompt": "Typically how much does an uber drive earn in a big uk city like Glasgow per day",
        "max_tokens": 36,
    },
    {
        "name": "keyboard",
        "prompt": "How much to send a yamaha keyboard from Dublin to Edinburgh",
        "max_tokens": 48,
    },
    {
        "name": "visas",
        "prompt": "Hey, can you just tell me about the new visas laws So, for example",
        "max_tokens": 6,
    },
    {
        "name": "vllm-lines",
        "prompt": "VLLm as a whole has many lines of code, how many is the python version",
        "max_tokens": 20,
    },
    {
        "name": "yamal",
        "prompt": "Is Lamine Yamal a prospect or a superstar",
        "max_tokens": 16,
    },
    {
        "name": "world-cup",
        "prompt": "I’m watching the World Cup. Who are the players to keep an eye on",
        "max_tokens": 28,
    },
    {
        "name": "google",
        "prompt": "how much does google pay for compute monthly",
        "max_tokens": 32,
    },
    {
        "name": "sleep",
        "prompt": "When fixing a sleep schedule how much time intervals should I bring back with",
        "max_tokens": 18,
    },
    {
        "name": "bitcoin",
        "prompt": "I don’t understand the concept of a bitcoin",
        "max_tokens": 22,
    },
    {
        "name": "ferguson",
        "prompt": "Which football manager influenced English football the most ?",
        "max_tokens": 12,
    },
]


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available. Run this on the RunPod GPU, not on a laptop.")
    try:
        import triton  # noqa: F401
    except ImportError:
        sys.exit(
            "triton is not installed, so the CUDA kernels would not run.\n"
            "Install it with: pip install triton"
        )


def esc(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("|", "/")
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def piece(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id])


def hex_ptr(ptr: int) -> str:
    return f"0x{ptr:x}"


def attention_caches(model):
    caches = []
    for layer in model.model.layers:
        caches.append(layer.self_attn.attn.kv_cache)
    return caches


def locate_slot(cache, slot: int, block_size: int, table: list[int], position: int) -> dict:
    phys, off = divmod(slot, block_size)
    logical, off_from_pos = divmod(position, block_size)
    k = cache[0, phys, off]
    v = cache[1, phys, off]
    k_ptr = int(k.data_ptr())
    v_ptr = int(v.data_ptr())
    return {
        "position": position,
        "slot": slot,
        "physical_block": phys,
        "offset": off,
        "logical_block": logical,
        "formula_ok": bool(
            logical < len(table)
            and table[logical] == phys
            and off == off_from_pos
            and slot == table[logical] * block_size + off
        ),
        "k_ptr": k_ptr,
        "v_ptr": v_ptr,
        "k_byte_offset": k_ptr - int(cache[0].data_ptr()),
        "v_byte_offset": v_ptr - int(cache[1].data_ptr()),
    }


def slot_k_rms(cache, slot: int, block_size: int) -> float:
    phys, off = divmod(slot, block_size)
    k = cache[0, phys, off].float()
    return float(k.pow(2).mean().sqrt())


class Probe:
    """Record scheduler output and whether the step replayed a CUDA graph."""

    def __init__(self, engine):
        self.engine = engine
        self.pending = None
        sched = engine.scheduler
        runner = engine.model_runner
        orig_schedule = sched.schedule
        orig_execute = runner.execute_model

        def schedule():
            mgr = sched.kv_cache_manager
            pre_free = mgr.num_free_blocks
            pre_head = list(mgr.free_block_ids)[:8]
            pre_used = set(mgr.used_block_ids)
            out = orig_schedule()
            tables = {
                seq.seq_id: list(out.block_tables[seq.seq_id]) for seq in out.scheduled_sequences
            }
            claimed = set()
            new_blocks = {}
            for seq in out.scheduled_sequences:
                grown = [b for b in tables[seq.seq_id] if b not in pre_used and b not in claimed]
                new_blocks[seq.seq_id] = grown
                claimed.update(grown)
            queries = {}
            for seq in out.scheduled_sequences:
                n = out.num_scheduled_tokens[seq.seq_id]
                start = seq.num_computed_tokens
                queries[seq.seq_id] = {
                    "start": start,
                    "n": n,
                    "token_ids": list(seq.token_ids[start : start + n]),
                }
            self.pending = {
                "scheduled_ids": [seq.seq_id for seq in out.scheduled_sequences],
                "preempted_ids": [seq.seq_id for seq in out.preempted_seqs],
                "free_before": pre_free,
                "free_after_alloc": mgr.num_free_blocks,
                "free_head_before": pre_head,
                "free_head_after_alloc": list(mgr.free_block_ids)[:8],
                "tables": tables,
                "slots": {
                    seq.seq_id: list(out.slot_mappings[seq.seq_id])
                    for seq in out.scheduled_sequences
                },
                "new_blocks": new_blocks,
                "queries": queries,
                "exec": None,
            }
            return out

        def execute(scheduler_output):
            import torch
            from babyvllm.worker.model_runner import BATCH_BUCKETS, CTX_BUCKETS, ceil_to_bucket

            graphed = runner._can_graph_decode(scheduler_output)
            bucket = None
            new_graph = False
            if graphed and scheduler_output.scheduled_sequences:
                seqs = scheduler_output.scheduled_sequences
                max_seq = max(seq.num_computed_tokens + 1 for seq in seqs)
                bb = ceil_to_bucket(len(seqs), BATCH_BUCKETS)
                cb = ceil_to_bucket(max_seq, CTX_BUCKETS)
                bucket = [bb, cb]
                new_graph = (bb, cb) not in runner._graphs
                if new_graph:
                    print(
                        f"  capturing CUDA graph batch_bucket={bb} ctx_bucket={cb} "
                        "(first use of this bucket; the next steps replay it)",
                        flush=True,
                    )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = orig_execute(scheduler_output)
            torch.cuda.synchronize()
            if self.pending is not None:
                self.pending["exec"] = {
                    "graphed": graphed,
                    "bucket": bucket,
                    "new_graph": new_graph,
                    "ms": (time.perf_counter() - t0) * 1000.0,
                    "mem_allocated": torch.cuda.memory_allocated(),
                    "mem_reserved": torch.cuda.memory_reserved(),
                }
            return result

        sched.schedule = schedule
        runner.execute_model = execute


def kv_layout(llm) -> dict:
    import torch

    runner = llm.model_runner
    caches = attention_caches(runner.model)
    cache = caches[0]
    cfg = llm.model_config
    props = torch.cuda.get_device_properties(0)
    profile = runner.kv_profile or {}
    total_bytes = sum(c.numel() * c.element_size() for c in caches)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_bytes": props.total_memory,
        "gpu_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "dtype": str(runner.dtype),
        "model": llm.model_name,
        "tokenizer": getattr(llm.tokenizer, "name_or_path", type(llm.tokenizer).__name__),
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "num_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "block_size": runner.block_size,
        "manager_blocks": llm.engine.kv_cache_manager.num_blocks,
        "dummy_block": runner._dummy_block,
        "cuda_graphs": runner.use_cuda_graphs,
        "tensor_blocks": cache.shape[1],
        "shape": list(cache.shape),
        "element_bytes": cache.element_size(),
        "k_stride_bytes": [s * cache.element_size() for s in cache[0].stride()],
        "layer0_base": int(cache.data_ptr()),
        "layer0_k_base": int(cache[0].data_ptr()),
        "layer0_v_base": int(cache[1].data_ptr()),
        "layer1_base": int(caches[1].data_ptr()) if len(caches) > 1 else None,
        "last_layer_base": int(caches[-1].data_ptr()),
        "dummy_k_ptr": int(cache[0, runner._dummy_block, 0].data_ptr()),
        "bytes_all_layers": total_bytes,
        "row_bytes": cfg.num_key_value_heads * cfg.head_dim * cache.element_size(),
        "profile": {k: (float(v) if isinstance(v, float) else v) for k, v in profile.items()},
        "mem_allocated_after_load": torch.cuda.memory_allocated(),
        "mem_reserved_after_load": torch.cuda.memory_reserved(),
    }


def attach_probe(engine) -> Probe:
    probe = Probe(engine)
    engine.scheduler.schedule._probe = probe
    return probe


def _pending(engine):
    return engine.scheduler.schedule._probe.pending


def run_jobs(llm, jobs: list[dict], detailed: bool) -> dict:
    import torch
    from babyvllm.sequence import SamplingParams

    engine = llm.engine
    engine.reset()
    caches = attention_caches(llm.model_runner.model)
    print("  zeroing KV cache so a written slot can be told from an empty one", flush=True)
    for cache in caches:
        cache.zero_()
    torch.cuda.synchronize()

    labels = {}
    prompt_ids = {}
    order = []
    for job in jobs:
        ids = llm.tokenizer(job["prompt"]).input_ids
        seq_id = engine.add_request(
            ids,
            SamplingParams(temperature=0.0, max_tokens=job["max_tokens"], ignore_eos=True),
        )
        labels[seq_id] = job["name"]
        prompt_ids[seq_id] = {
            "name": job["name"],
            "prompt": job["prompt"],
            "max_tokens": job["max_tokens"],
            "token_ids": ids,
            "pieces": [piece(llm.tokenizer, t) for t in ids],
        }
        order.append(seq_id)
        print(
            f"  queued {job['name']} seq={seq_id} prompt_tokens={len(ids)} "
            f"max_new={job['max_tokens']}",
            flush=True,
        )

    cache = caches[0]
    block_size = llm.model_runner.block_size
    steps = []
    generated = {seq_id: [] for seq_id in order}
    step_no = 0
    while engine.has_unfinished_requests():
        step_no += 1
        outputs = engine.step()
        snap = _pending(engine)
        mgr = engine.kv_cache_manager
        by_id = {out.seq_id: out for out in outputs}
        rows = []
        for seq_id in snap["scheduled_ids"]:
            q = snap["queries"][seq_id]
            slots = snap["slots"][seq_id]
            table = snap["tables"][seq_id]
            infos = [
                locate_slot(cache, slot, block_size, table, q["start"] + i)
                for i, slot in enumerate(slots)
            ]
            if detailed and infos:
                infos[0]["k_rms"] = slot_k_rms(cache, infos[0]["slot"], block_size)
                if len(infos) > 1:
                    infos[-1]["k_rms"] = slot_k_rms(cache, infos[-1]["slot"], block_size)
            out = by_id.get(seq_id)
            sampled = None
            sampled_piece = None
            finished = False
            reason = None
            if out is not None:
                if out.new_token_ids:
                    sampled = out.new_token_ids[-1]
                    sampled_piece = piece(llm.tokenizer, sampled)
                    generated[seq_id].extend(out.new_token_ids)
                finished = bool(out.finished)
                if out.finish_reason is not None:
                    reason = str(out.finish_reason.value)
            occupied = q["start"] + q["n"]
            last_fill = occupied - (len(table) - 1) * block_size
            rows.append(
                {
                    "seq_id": seq_id,
                    "name": labels[seq_id],
                    "phase": "prefill" if q["start"] == 0 else "decode",
                    "position_start": q["start"],
                    "position_end": q["start"] + q["n"] - 1,
                    "query_token_ids": q["token_ids"],
                    "query_pieces": [piece(llm.tokenizer, t) for t in q["token_ids"]],
                    "block_table": table,
                    "new_blocks": snap["new_blocks"][seq_id],
                    "last_block_fill": last_fill,
                    "slots": slots,
                    "slot_info": infos,
                    "sampled_id": sampled,
                    "sampled_piece": sampled_piece,
                    "finished": finished,
                    "finish_reason": reason,
                }
            )
        post_alloc_used = set()
        for row in rows:
            post_alloc_used.update(row["block_table"])
        freed = sorted(post_alloc_used - set(mgr.used_block_ids))
        steps.append(
            {
                "step": step_no,
                "free_before": snap["free_before"],
                "free_after_alloc": snap["free_after_alloc"],
                "free_after_step": mgr.num_free_blocks,
                "free_head_before": snap["free_head_before"],
                "free_head_after_step": list(mgr.free_block_ids)[:8],
                "freed_blocks": freed,
                "preempted_ids": [labels.get(i, i) for i in snap["preempted_ids"]],
                "exec": snap["exec"],
                "batch": len(rows),
                "rows": rows,
            }
        )
        names = ",".join(
            f"{r['name']}:{r['phase'][0].upper()}{r['position_end']}"
            + ("*" if r["new_blocks"] else "")
            + ("!" if r["finished"] else "")
            for r in rows
        )
        exec_info = snap["exec"] or {}
        mode = "graph" if exec_info.get("graphed") else "eager"
        print(
            f"  step {step_no:03d} batch={len(rows)} {mode} "
            f"free {snap['free_before']}->{mgr.num_free_blocks} {names}",
            flush=True,
        )

    texts = {}
    for seq_id, ids in generated.items():
        texts[labels[seq_id]] = {
            "seq_id": seq_id,
            "token_ids": ids,
            "text": llm.tokenizer.decode(ids),
        }
    return {
        "prompts": {str(k): v for k, v in prompt_ids.items()},
        "steps": steps,
        "generated": texts,
        "free_at_end": engine.kv_cache_manager.num_free_blocks,
        "manager_blocks": engine.kv_cache_manager.num_blocks,
        "layer0_base_after": int(caches[0].data_ptr()),
    }


def gb(n: int | float) -> str:
    return f"{n / 1e9:.2f} GB"


def render_prompt_block(prompt: dict) -> list[str]:
    lines = [
        f"### {prompt['name']}",
        "",
        f"Prompt: `{prompt['prompt']}`",
        "",
        f"Qwen token ids ({len(prompt['token_ids'])}), in order:",
        "",
        "| pos | token id | piece |",
        "| ---: | ---: | --- |",
    ]
    for i, (tid, text) in enumerate(zip(prompt["token_ids"], prompt["pieces"])):
        lines.append(f"| {i} | {tid} | `{esc(text)}` |")
    lines.append("")
    lines.append(f"Generation cap: {prompt['max_tokens']} new tokens, greedy (`temperature=0`), EOS ignored.")
    lines.append("")
    return lines


def render_slot_table(infos: list[dict], limit: int | None = None) -> list[str]:
    shown = infos if limit is None else infos[:limit]
    lines = [
        "| pos | slot | block | off | formula ok | layer0 K | layer0 V | K byte offset | K rms |",
        "| ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for info in shown:
        rms = info.get("k_rms")
        rms_s = "" if rms is None else f"{rms:.4f}"
        lines.append(
            f"| {info['position']} | {info['slot']} | {info['physical_block']} | {info['offset']} | "
            f"{info['formula_ok']} | `{hex_ptr(info['k_ptr'])}` | `{hex_ptr(info['v_ptr'])}` | "
            f"{info['k_byte_offset']} | {rms_s} |"
        )
    if limit is not None and len(infos) > limit:
        lines.append(f"| … | {len(infos) - limit} more slots in the JSON | | | | | | | |")
    lines.append("")
    return lines


def render_phase(title: str, phase: dict, detailed: bool) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append(
        "Generation is greedy and stops at each request's `max_tokens` cap, with EOS ignored. "
        "The caps are short so a short request can finish and return its blocks while a longer one is still running. "
        "These are not the long chat replies."
    )
    lines.append("")
    lines.append("### Token ids")
    lines.append("")
    for prompt in phase["prompts"].values():
        lines.extend(render_prompt_block(prompt))

    lines.append("### What the scheduler did")
    lines.append("")
    lines.append(
        "A `*` on a row means that step popped a new KV block. A `!` means the request finished on that step and its blocks were appended back onto the free list."
    )
    lines.append("")
    lines.append(
        "| step | batch | path | ms | free before | free after | who |"
    )
    lines.append("| ---: | ---: | --- | ---: | ---: | ---: | --- |")
    for step in phase["steps"]:
        exec_info = step["exec"] or {}
        if exec_info.get("graphed"):
            bucket = exec_info.get("bucket")
            path = f"CUDA graph {bucket}"
            if exec_info.get("new_graph"):
                path += " capture"
            else:
                path += " replay"
        else:
            path = "eager"
        who = ", ".join(
            f"{r['name']} {r['phase']} pos {r['position_start']}-{r['position_end']} "
            f"table {r['block_table']}"
            f"{' new ' + str(r['new_blocks']) if r['new_blocks'] else ''}"
            f"{' FINISH freed ' + str([b for b in step['freed_blocks'] if b in r['block_table']]) if r['finished'] else ''}"
            f" → {r['sampled_id']} `{esc(r['sampled_piece'] or '')}`"
            for r in step["rows"]
        )
        ms = exec_info.get("ms")
        ms_s = "" if ms is None else f"{ms:.1f}"
        lines.append(
            f"| {step['step']} | {step['batch']} | {path} | {ms_s} | "
            f"{step['free_before']} | {step['free_after_step']} | {who} |"
        )
    lines.append("")

    if any(step["preempted_ids"] for step in phase["steps"]):
        lines.append("Preemption happened:")
        for step in phase["steps"]:
            if step["preempted_ids"]:
                lines.append(f"- step {step['step']}: {step['preempted_ids']}")
        lines.append("")
    else:
        lines.append("No request was preempted. The free list had room for every allocation.")
        lines.append("")

    lines.append(f"Free blocks at the end: {phase['free_at_end']} of {phase['manager_blocks']}.")
    lines.append("")

    if detailed:
        lines.append("### Slot addresses")
        lines.append("")
        lines.append(
            "Layer 0 only. Every other layer has its own tensor and the same slot index. "
            "`K rms` is the root-mean-square of the bf16 key vector at that slot after the step. "
            "The cache was zeroed first, so a non-zero value means this step wrote it."
        )
        lines.append("")
        for step in phase["steps"]:
            for row in step["rows"]:
                interesting = (
                    step["step"] == 1
                    or row["new_blocks"]
                    or row["finished"]
                    or row["phase"] == "prefill"
                    or step["step"] <= 3
                )
                if not interesting:
                    continue
                lines.append(
                    f"#### Step {step['step']} — {row['name']} {row['phase']} "
                    f"positions {row['position_start']}-{row['position_end']}"
                )
                lines.append("")
                lines.append(
                    f"Block table `{row['block_table']}`, new blocks `{row['new_blocks']}`, "
                    f"last block fill {row['last_block_fill']}, "
                    f"sampled token {row['sampled_id']} `{esc(row['sampled_piece'] or '')}`."
                )
                lines.append("")
                lines.extend(render_slot_table(row["slot_info"]))

    lines.append("### Text the model actually produced")
    lines.append("")
    for name, gen in phase["generated"].items():
        lines.append(f"**{name}** seq {gen['seq_id']}, {len(gen['token_ids'])} new token ids:")
        lines.append("")
        lines.append("`" + " ".join(str(t) for t in gen["token_ids"]) + "`")
        lines.append("")
        lines.append(gen["text"] if gen["text"] else "(empty)")
        lines.append("")
    return lines


def render_memory(layout: dict, pair_base: int, batch_base: int) -> list[str]:
    profile = layout["profile"]
    lines = [
        "## Where the KV cache sits in GPU memory",
        "",
        f"GPU: {layout['gpu_name']} (compute capability {layout['gpu_capability']}), {gb(layout['gpu_total_bytes'])} total.",
        f"PyTorch {layout['torch']}, dtype `{layout['dtype']}`.",
        f"Model `{layout['model']}`, tokenizer `{layout['tokenizer']}`, vocab {layout['vocab_size']}.",
        "",
        f"Architecture read from the checkpoint: {layout['num_layers']} layers, "
        f"{layout['num_heads']} query heads, {layout['num_kv_heads']} KV heads, head dim {layout['head_dim']}.",
        "",
        f"Each attention layer owns one tensor of shape `{layout['shape']}` "
        f"(K/V, blocks, block size, KV heads, head dim), {layout['element_bytes']} bytes per element.",
        f"There are {layout['num_layers']} of those tensors, {gb(layout['bytes_all_layers'])} together.",
        "",
        f"Layer 0 base `{hex_ptr(layout['layer0_base'])}`.",
        f"Layer 0 K base `{hex_ptr(layout['layer0_k_base'])}`, V base `{hex_ptr(layout['layer0_v_base'])}`.",
        f"Layer 1 base `{hex_ptr(layout['layer1_base'])}`.",
        f"Last layer base `{hex_ptr(layout['last_layer_base'])}`.",
        "",
        f"The block manager was given {layout['manager_blocks']} blocks of size {layout['block_size']}.",
        f"The tensor holds {layout['tensor_blocks']} blocks. "
        f"CUDA graphs are {'on' if layout['cuda_graphs'] else 'off'}. "
        f"Dummy block id is {layout['dummy_block']}, and its first K slot is `{hex_ptr(layout['dummy_k_ptr'])}`. "
        "The manager never hands that block to a request. Graph replay pads unused batch rows with it.",
        "",
        f"One K row (one token, one layer) is {layout['row_bytes']} bytes. "
        f"K strides in bytes are `{layout['k_stride_bytes']}` for (block, offset, kv head, dim).",
        "",
        f"After load, PyTorch reported {gb(layout['mem_allocated_after_load'])} allocated and "
        f"{gb(layout['mem_reserved_after_load'])} reserved.",
        "",
        "KV profile from `allocate_kv_cache`:",
        "",
        "```",
        json.dumps(profile, indent=2, default=str),
        "```",
        "",
        f"Layer 0 base during the pair run `{hex_ptr(pair_base)}`, during the batch run `{hex_ptr(batch_base)}`. "
        + (
            "Same address: `reset()` returns blocks to the free list and does not reallocate the tensors."
            if pair_base == batch_base
            else "The address changed between runs."
        ),
        "",
    ]
    return lines


def render(layout: dict, pair: dict, batch: dict) -> str:
    formula_fail = 0
    formula_n = 0
    nonzero_writes = 0
    zero_writes = 0
    for step in pair["steps"]:
        for row in step["rows"]:
            for info in row["slot_info"]:
                formula_n += 1
                if not info["formula_ok"]:
                    formula_fail += 1
                if "k_rms" in info:
                    if info["k_rms"] > 0:
                        nonzero_writes += 1
                    else:
                        zero_writes += 1
    lines = [
        "# Baby-vLLM batch trace",
        "",
        "Every number below was recorded by `scripts/trace_batch_on_gpu.py` in this process. "
        "Token ids are the Qwen2.5 tokenizer and the greedy samples from this model. "
        "Addresses are `data_ptr()` on the live KV tensors.",
        "",
        f"Slot check: {formula_n - formula_fail} of {formula_n} pair slots matched "
        f"`physical_block * block_size + offset`. Failures: {formula_fail}.",
        f"Write check: {nonzero_writes} sampled key rows were non-zero after the step, {zero_writes} were still zero.",
        "",
    ]
    lines.extend(render_memory(layout, pair["layer0_base_after"], batch["layer0_base_after"]))
    lines.extend(
        render_phase(
            "Two sequences, full lifecycle",
            pair,
            detailed=True,
        )
    )
    lines.extend(
        render_phase(
            "Batch of ten prompts",
            batch,
            detailed=False,
        )
    )
    lines.append("## How a step is put together")
    lines.append("")
    lines.append(
        "Step 1 of each run is prefill. Every waiting request whose prompt fits in the token budget is admitted in one forward. "
        "The input is the concatenation of those prompts, not a rectangle padded to the longest one. "
        "That is selective batching."
    )
    lines.append("")
    lines.append(
        "Later steps are decode: one new token per request that is still running. "
        "When a request hits its cap it is removed and its blocks go back on the free list before the next step. "
        "The next forward has a smaller batch. That is continuous batching."
    )
    lines.append("")
    lines.append(
        "A pure decode step can replay a CUDA graph. The graph is keyed by a batch bucket "
        "(1, 2, 4, 8, 16, 32, 64) and a context bucket (256 or 2048). "
        "The first step that needs a bucket captures it, which is why that step's time is much larger. "
        "A step that still contains a prefill stays on the eager path."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    require_cuda()
    import torch
    from babyvllm import LLM

    model = os.environ.get("BABYVLLM_MODEL", "Qwen/Qwen2.5-7B")
    out_dir = Path(os.environ.get("TRACE_OUT", "trace_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Loading {model}", flush=True)
    llm = LLM(model, device="cuda", verbose=True)
    attach_probe(llm.engine)
    layout = kv_layout(llm)
    print(
        f"KV tensors: {layout['num_layers']} x {layout['shape']} "
        f"manager_blocks={layout['manager_blocks']} dummy={layout['dummy_block']}",
        flush=True,
    )

    print("\n=== pair ===", flush=True)
    pair = run_jobs(llm, PAIR, detailed=True)
    print("\n=== batch ===", flush=True)
    batch = run_jobs(llm, BATCH, detailed=False)

    payload = {"layout": layout, "pair": pair, "batch": batch}
    json_path = out_dir / "batch_trace.json"
    md_path = out_dir / "batch_trace.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    md_path.write_text(render(layout, pair, batch))
    print(f"\nWrote {md_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
