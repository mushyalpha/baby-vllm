import importlib.util
import sys
from pathlib import Path

import torch

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"


def _load():
    sys.path.insert(0, str(BENCH_DIR))
    spec = importlib.util.spec_from_file_location("profile_decode_host", BENCH_DIR / "profile_decode_host.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_attributes_time_to_phases_and_restores_the_engine():
    from test_benchmark_prefix_caching import _tiny_engine

    mod = _load()
    engine, _ = _tiny_engine()
    schedule_before = engine.scheduler.schedule
    rows, _, prepare_name = mod.run_trial(
        engine, [11, 12, 13, 14, 15, 16, 17, 18], batch=4, warmup=2, steps=5,
        device=torch.device("cpu"), cprofile=False,
    )
    assert len(rows) == 5
    assert engine.scheduler.schedule == schedule_before
    assert prepare_name in ("prepare_decode_inputs", "prepare_static_inputs")
    for row in rows:
        assert row["schedule"] > 0 and row["update"] > 0
        assert row["wall"] >= row["schedule"] + row["update"]
        assert "graph_gpu" not in row  # no events off CUDA
    assert set(mod.summarize(rows)) >= {"wall", "schedule", "prepare", "launch", "update", "rest"}
