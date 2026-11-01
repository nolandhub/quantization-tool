# ABOUTME: Tests for qtool.plan: runs dir resolution, checkpoint reuse/stale decisions from qtool.json,
# ABOUTME: baseline cache validity with readable stale reasons, and the plan text.
import json

from qtool import plan


def sig(**over):
    base = plan.checkpoint_signature("Qwen/Qwen3.5-4B", "INT4-W4A16", "HuggingFaceH4/ultrachat_200k:train_sft", 512,
                                     2048, [], "0.13.0")
    return {**base, **over}


def write_ckpt(rd, name, signature):
    d = rd / "checkpoints" / name
    d.mkdir(parents=True)
    (d / "qtool.json").write_text(json.dumps({"signature": signature}))


def test_runs_dir(tmp_path):
    assert plan.runs_dir(tmp_path, "Qwen/Qwen3.5-4B") == tmp_path / "Qwen3.5-4B"
    ck = tmp_path / "Qwen3.5-4B" / "checkpoints" / "INT4-W4A16"
    ck.mkdir(parents=True)
    assert plan.runs_dir(tmp_path / "elsewhere", str(ck)) == tmp_path / "Qwen3.5-4B"


def test_checkpoint_states(tmp_path):
    rd = tmp_path / "m"
    write_ckpt(rd, "INT4-W4A16", sig())
    write_ckpt(rd, "INT8-W8A8", sig(checkpoint="INT8-W8A8", calib_samples=256))
    wanted = {"INT4-W4A16": sig(), "INT8-W8A8": sig(checkpoint="INT8-W8A8"), "X": sig(checkpoint="X")}
    needed = {"INT4-W4A16": ["int4-w4a16"], "INT8-W8A8": ["int8-w8a8"], "X": ["x"]}
    states = plan.checkpoint_states(rd, needed, wanted, force=False)
    assert [(s.name, s.action, s.diff) for s in states] == [
        ("INT4-W4A16", "reuse", []), ("INT8-W8A8", "stale", ["calib_samples: 256 → 512"]), ("X", "build", [])]
    assert plan.checkpoint_states(rd, {"INT8-W8A8": ["int8-w8a8"]}, wanted, force=True)[0].action == "rebuild"


def test_checkpoint_states_unreadable_meta(tmp_path):
    rd = tmp_path / "m"
    d = rd / "checkpoints" / "INT4-W4A16"
    d.mkdir(parents=True)
    (d / "qtool.json").write_text("{truncated")
    needed, wanted = {"INT4-W4A16": ["int4-w4a16"]}, {"INT4-W4A16": sig()}
    states = plan.checkpoint_states(rd, needed, wanted, force=False)
    assert [(s.action, s.diff) for s in states] == [("stale", ["qtool.json unreadable: JSONDecodeError"])]
    assert [s.action for s in plan.checkpoint_states(rd, needed, wanted, force=True)] == ["rebuild"]


def test_checkpoint_states_meta_is_directory(tmp_path):
    rd = tmp_path / "m"
    (rd / "checkpoints" / "INT4-W4A16" / "qtool.json").mkdir(parents=True)
    states = plan.checkpoint_states(rd, {"INT4-W4A16": ["int4-w4a16"]}, {"INT4-W4A16": sig()}, force=False)
    assert [(s.action, s.diff) for s in states] == [("stale", ["qtool.json unreadable: IsADirectoryError"])]


def test_cached_result(tmp_path):
    p = tmp_path / "bf16_gsm8k.json"
    wanted = plan.eval_meta("Qwen/Qwen3.5-4B", "gsm8k", None, "qtool-eval:abc", "chat", "RTX 4070")
    assert plan.cached_result(p, wanted) == (False, "cache: none, will run")
    p.write_text(json.dumps({"qtool": {**wanted, "limit": 250}}))
    assert plan.cached_result(p, wanted) == (False, "cache stale: eval-limit: 250 → full, will run")
    p.write_text(json.dumps({"qtool": wanted}))
    assert plan.cached_result(p, wanted) == (True, "cache: valid, reused")


def test_cached_result_unreadable(tmp_path):
    p = tmp_path / "bf16_gsm8k.json"
    p.write_text("{truncated")
    wanted = plan.eval_meta("Qwen/Qwen3.5-4B", "gsm8k", None, "qtool-eval:abc", "chat", "RTX 4070")
    assert plan.cached_result(p, wanted) == (False, "cache unreadable, will run")


def test_calib_key_is_stable_and_sensitive():
    k = plan.calib_key("m", "c", 512, 2048)
    assert k == plan.calib_key("m", "c", 512, 2048) and len(k) == 12 and k != plan.calib_key("m", "c", 256, 2048)


def test_render_plan():
    states = [plan.CheckpointState("INT4-W4A16", ["int4-w4a16", "int4-w4a8-fp8"], "build", []),
              plan.CheckpointState("INT8-W8A8", ["int8-w8a8"], "reuse", [])]
    lines = plan.render_plan("Qwen/Qwen3.5-4B", states, ["int4-w4a16", "int4-w4a8-fp8", "int8-w8a8", "fp8-w8a8"],
                             ["gsm8k"], None, "cache: none, will run", no_eval=False)
    assert lines == [
        "Plan for Qwen/Qwen3.5-4B (4 schemes)",
        "  quantize  INT4-W4A16  -> shared by: int4-w4a16, int4-w4a8-fp8",
        "  reuse     INT8-W8A8   -> checkpoints/INT8-W8A8 (qtool.json matches)",
        "  online    fp8-w8a8    -> vLLM --quantization fp8 (no checkpoint)",
        "  eval      bf16 baseline (cache: none, will run)",
        "  eval      int4-w4a16, int4-w4a8-fp8, int8-w8a8, fp8-w8a8   tasks: gsm8k (full)",
    ]
    assert plan.render_plan("m", [], ["fp8-w8a8"], ["gsm8k"], 50, None, no_eval=True)[-1] == "  eval      skipped (--no-eval)"
