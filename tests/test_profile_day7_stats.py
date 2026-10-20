import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "profile_day7.py"
    spec = importlib.util.spec_from_file_location("profile_day7", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summarize_trials_ci_uses_sample_std():
    mod = _load()
    stats = mod.summarize_trials([1.0, 2.0, 3.0])
    assert stats["n"] == 3
    assert stats["mean"] == 2.0
    assert stats["std"] == 1.0
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert abs(stats["ci95"] - 1.96 / (3 ** 0.5)) < 1e-12


def test_summarize_trials_single_run_has_no_interval():
    mod = _load()
    stats = mod.summarize_trials([5.5])
    assert stats["std"] == 0.0
    assert stats["ci95"] == 0.0
    assert stats["mean"] == 5.5


def test_fit_decode_window_leaves_short_context_alone():
    mod = _load()
    prompt, warmup, steps, note = mod.fit_decode_window(128, 30, 100, 2048)
    assert (prompt, warmup, steps, note) == (128, 30, 100, None)
    assert prompt + warmup + steps <= 2048


def test_fit_decode_window_shortens_steps_at_1920():
    mod = _load()
    prompt, warmup, steps, note = mod.fit_decode_window(1920, 30, 100, 2048)
    assert prompt == 1920
    assert warmup == 30
    assert steps == 98
    assert note
    assert prompt + warmup + steps == 2048


def test_fit_decode_window_backs_off_a_2048_prompt():
    mod = _load()
    prompt, warmup, steps, note = mod.fit_decode_window(2048, 30, 100, 2048)
    assert warmup == 30
    assert steps == 100
    assert prompt == 1918
    assert note
    assert prompt + warmup + steps == 2048


def test_clean_wall_reads_timing_pass_p50(tmp_path):
    mod = _load()
    path = tmp_path / "timing.json"
    path.write_text(json.dumps({"p50_mean": 5.84, "p50_ci95": 0.02}))
    clean = mod.clean_wall_from_timing(str(path))
    assert clean["wall_p50_ms"] == 5.84
    assert clean["ci95"] == 0.02
