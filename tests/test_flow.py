# ABOUTME: Tests for qtool.flow with a fake docker that replays real spike logs and writes the files the real
# ABOUTME: containers would (checkpoint dir, lm_eval results): headers, events, cache reuse, failures, --on-error, summary.
import io
import json

import pytest

from fakes import GPU, W8A8_QC, FakeDocker, arg, ev
from qtool import flow, plan
from qtool.console import Console

@pytest.fixture
def setup(fixtures, tmp_path):
    rd = tmp_path / "runs" / "Qwen3.5-4B"
    out = io.StringIO()
    clock = iter(range(0, 10_000_000, 5))
    s = flow.Session(Console(out, tty=False, wallclock=lambda: "21:00:00", clock=lambda: 0.0), GPU, rd,
                     tmp_path / "hf", tmp_path / "home", "1002:1002", "qtool-compress:cu129-abc", "qtool-eval:def",
                     25_000_000_000, clock=lambda: next(clock))
    docker = FakeDocker(fixtures, rd)
    s.run = docker
    docker.chowns = []
    s.chown = lambda argv, **kw: docker.chowns.append(argv)
    docker.evals = {"bf16": ("lm_eval/eval-bf16-chat.log", "lm_eval/out/bf16-chat", 0),
                    "int4-w4a16": ("lm_eval/eval-w4a16-none.log", "lm_eval/out/w4a16-none", 0),
                    "int4-w4a8-fp8": ("lm_eval/eval-w4a16-fp8.log", "lm_eval/out/w4a16-fp8", 0),
                    "nvfp4-w4a4": ("lm_eval/eval-w4a16-none.log", "lm_eval/out/w4a16-none", 0),
                    "int8-w8a8": ("lm_eval/eval-w8a8.log", "lm_eval/out/w8a8", 0),
                    "fp8-w8a8": ("lm_eval/eval-fp8-online.log", "lm_eval/out/fp8-online", 0)}
    return s, docker, out


def quantize_run(s, names, no_eval=False, on_error="continue"):
    needed = flow.schemes.checkpoints_for(names)
    sigs = {c: plan.checkpoint_signature("Qwen/Qwen3.5-4B", c, "HuggingFaceH4/ultrachat_200k:train_sft", 512, 2048, [],
                                         "0.13.0", "bfloat16") for c in needed}
    states = plan.checkpoint_states(s.rd, needed, sigs, force=False)
    opts = flow.QuantizeOptions("Qwen/Qwen3.5-4B", "HuggingFaceH4/ultrachat_200k:train_sft", 512, 2048, [], "bfloat16",
                                9_078_531_072)
    return flow.QuantizeRun("Qwen/Qwen3.5-4B", names, states, sigs, opts, 4_539_265_536, ["gsm8k"], 50, "chat",
                            no_eval, on_error, {"model": "Qwen/Qwen3.5-4B"})


def test_quantize_and_eval_two_schemes_sharing_a_checkpoint(setup):
    s, docker, out = setup
    code = flow.run_quantize(s, quantize_run(s, ["int4-w4a16", "int4-w4a8-fp8"]))
    text = out.getvalue()
    assert code == 0, text
    for h in ["[1/4] Loading model · INT4-W4A16", "[2/4] Preparing calibration data", "[3/4] Quantizing INT4-W4A16",
              "[4/4] Evaluating bf16 baseline", "[4/4] Evaluating int4-w4a16", "[4/4] Evaluating int4-w4a8-fp8"]:
        assert f"══════ {h} " in text
    assert "  architecture : Qwen3_5ForConditionalGeneration (verified) · multimodal" in text
    assert "  ignore       : lm_head (output head) · visual (vision/projector) · linear_attn (hybrid linear-attention / SSM mixer)" in text
    assert "  quantized    : 128 Linear (mlp.*, self_attn.*)" in text
    assert "  tokens/sample: min 199 · median 1,080 · max 2,048 (truncated: 48)" in text
    assert "  saved        : checkpoints/INT4-W4A16 · 9.1 GB → 5.2 GB · MTP saved" in text
    assert "  kernel       : MarlinLinearKernel for CompressedTensorsWNA16" in text
    assert "  runtime env  : VLLM_MARLIN_INPUT_DTYPE=fp8" in text
    assert "  checkpoint   : checkpoints/INT4-W4A16 (shared)" in text
    assert "  quantization : compressed-tensors · INT4 W4A16 g128 symmetric" in text
    assert "  vs bf16      : gsm8k strict-match 0.820 → 0.780 · −4.0 pp · recovery 95.1%" in text
    assert "  int4-w4a8-fp8   OK       0.780 ± 0.059   −4.0 pp     95.1%      5.2 GB (shared)" in text
    assert (s.rd / "checkpoints" / "INT4-W4A16" / "qtool.json").exists()
    assert not list(s.rd.rglob(".tmp-*"))
    quant = [sp for sp in docker.specs if "/qtool/quantize.py" in sp.command]
    assert len(quant) == 1 and quant[0].memory_bytes == 25_000_000_000 and quant[0].gpu == 0
    assert len(docker.chowns) == len(docker.specs)
    assert docker.chowns[0] == ["docker", "run", "--rm", "-v", f"{s.rd}:/runs", "-v", f"{s.home}:/qtool-home",
                                "--entrypoint", "chown", "qtool-compress:cu129-abc", "-R", "1002:1002", "/runs", "/qtool-home"]
    w4a8 = [sp for sp in docker.specs if sp.command[0] == "lm_eval" and sp.env.get("VLLM_MARLIN_INPUT_DTYPE") == "fp8"]
    assert len(w4a8) == 1 and "--apply_chat_template" in w4a8[0].command
    assert "pretrained=/runs/checkpoints/INT4-W4A16," in arg(w4a8[0].command, "--model_args")
    saved = json.loads(plan.result_path(s.rd, "bf16", "gsm8k").read_text())
    assert saved["qtool"]["mode"] == "chat" and saved["results"]["gsm8k"]["exact_match,strict-match"] == 0.82
    assert json.loads((s.rd / "results" / "summary.json").read_text())["rows"][0]["label"] == "bf16"


def test_headers_come_once_and_in_order(setup):
    s, docker, out = setup
    flow.run_quantize(s, quantize_run(s, ["int4-w4a16"], no_eval=True))
    headers = [line.split(" ═")[0].removeprefix("══════ ") for line in out.getvalue().splitlines() if line.startswith("══")]
    assert headers == ["[1/4] Loading model · INT4-W4A16", "[2/4] Preparing calibration data",
                       "[3/4] Quantizing INT4-W4A16", "Summary · checkpoints"]
    after_oneshot = out.getvalue().split("  oneshot      :")[1]
    assert "Propagating" not in after_oneshot


def test_second_run_reuses_checkpoint_and_baseline(setup):
    s, docker, out = setup
    flow.run_quantize(s, quantize_run(s, ["int4-w4a16"]))
    docker.specs.clear()
    code = flow.run_quantize(s, quantize_run(s, ["int4-w4a16"]))
    assert code == 0
    assert [sp.command[0] for sp in docker.specs] == ["lm_eval"]  # only the scheme eval runs again
    assert "  cache        : gsm8k: valid, reused" in out.getvalue()


def test_quantize_failure_at_step3_skips_dependent_schemes_and_continues(setup):
    s, docker, out = setup
    docker.quantize["INT4-W4A16"] = "fail-step3"
    code = flow.run_quantize(s, quantize_run(s, ["int4-w4a16", "int4-w4a8-fp8", "int8-w8a8"]))
    text = out.getvalue()
    assert code == 1
    assert "  ✗ FAILED: RuntimeError: Failed to find C compiler" in text
    assert "  int4-w4a8-fp8   SKIPPED  —               n/a         n/a        unknown (INT4-W4A16 failed)" in text
    assert "  int8-w8a8       OK" in text
    assert not (s.rd / "checkpoints" / ".tmp-INT4-W4A16").exists()


def test_failure_before_step3_stops_everything(setup):
    s, docker, out = setup
    docker.quantize["INT4-W4A16"] = "fail-step2"
    code = flow.run_quantize(s, quantize_run(s, ["int4-w4a16", "int8-w8a8"]))
    text = out.getvalue()
    assert code == 1
    assert "stopping all schemes: model loading / calibration is shared by every checkpoint" in text
    assert "DatasetNotFoundError: no such dataset" in text
    assert [sp.command[0] for sp in docker.specs] == ["python3"]
    assert "  INT8-W8A8   NOT RUN" in text


def test_killed_container_reports_host_ram(setup):
    s, docker, out = setup
    docker.quantize["INT4-W4A16"] = "killed"
    assert flow.run_quantize(s, quantize_run(s, ["int4-w4a16"], no_eval=True)) == 1
    assert "container killed (exit 137): host RAM ran out at the --memory limit of 25.0 GB" in out.getvalue()


def test_on_error_stop(setup):
    s, docker, out = setup
    docker.quantize["INT4-W4A16"] = "fail-step3"
    assert flow.run_quantize(s, quantize_run(s, ["int4-w4a16", "int8-w8a8"], on_error="stop")) == 1
    assert sum(1 for sp in docker.specs if "/qtool/quantize.py" in sp.command) == 1


def test_eval_failure_quotes_the_vllm_error(setup):
    s, docker, out = setup
    docker.evals["int8-w8a8"] = ("lm_eval/eval-bf16-docs.err-mamba.log", None, 1)
    assert flow.run_quantize(s, quantize_run(s, ["int8-w8a8", "fp8-w8a8"])) == 1
    text = out.getvalue()
    assert "max_num_seqs (256) exceeds available Mamba cache blocks (124)" in text
    assert "  int8-w8a8       FAILED" in text and "  fp8-w8a8        OK" in text


def test_fp8_checkpoint_is_built_without_calibration_and_evaluated(setup):
    s, docker, out = setup
    assert flow.run_quantize(s, quantize_run(s, ["fp8-w8a8"])) == 0, out.getvalue()
    text = out.getvalue()
    assert "══════ [2/4] Preparing calibration data " in text
    assert "  calibration  : not needed: weights are rounded, activation scales are computed at runtime" in text
    assert "══════ [3/4] Quantizing FP8-DYNAMIC " in text
    assert "  recipe       : QuantizationModifier targets=Linear scheme=FP8_DYNAMIC\n" in text
    assert "  quantization : compressed-tensors · FLOAT8 W8A8 symmetric" in text
    assert "  checkpoint   : checkpoints/FP8-DYNAMIC" in text
    assert (s.rd / "checkpoints" / "FP8-DYNAMIC" / "qtool.json").exists()
    fp8 = [sp for sp in docker.specs if sp.command[0] == "lm_eval" and "/runs/checkpoints/FP8-DYNAMIC" in arg(sp.command, "--model_args")]
    assert len(fp8) == 1 and "quantization=" not in arg(fp8[0].command, "--model_args")


def test_nvfp4_checkpoint_is_quantized_and_evaluated(setup):
    s, docker, out = setup
    assert flow.run_quantize(s, quantize_run(s, ["nvfp4-w4a4"])) == 0
    text = out.getvalue()
    assert "[3/4] Quantizing NVFP4" in text
    assert "  checkpoint   : checkpoints/NVFP4" in text and "(shared)" not in text
    assert "  quantization : compressed-tensors · NVFP4 W4A4 g16" in text
    assert "  nvfp4-w4a4" in text and "VLLM_MARLIN_INPUT_DTYPE" not in text


def test_nvfp4_quantization_text_without_input_activations():
    qc = {"quant_method": "compressed-tensors",
          "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}}
    assert flow.quantization_text(qc) == "compressed-tensors · NVFP4 W4A16 g16"


def test_w4a8_without_marlin_line_is_invalid(setup):
    s, docker, out = setup
    docker.evals["int4-w4a8-fp8"] = ("lm_eval/eval-bf16-chat.log", "lm_eval/out/bf16-chat", 0)
    assert flow.run_quantize(s, quantize_run(s, ["int4-w4a8-fp8"])) == 1
    assert "INVALID: no MarlinLinearKernel line" in out.getvalue()


def test_baseline_that_does_not_fit_is_skipped(setup):
    s, docker, out = setup
    r = quantize_run(s, ["int4-w4a16"])
    r.params = 7_000_000_000  # 14 GB BF16 > 13.5 GB budget on 16 GB
    assert flow.run_quantize(s, r) == 0
    text = out.getvalue()
    assert "bf16 baseline does not fit: 14.0 GB weights > 13.5 GB budget" in text
    assert "  int4-w4a16      OK       0.760 ± 0.061   n/a         n/a" in text


def test_no_eval_prints_checkpoint_summary(setup):
    s, docker, out = setup
    assert flow.run_quantize(s, quantize_run(s, ["int4-w4a16"], no_eval=True)) == 0
    assert "  INT4-W4A16  OK       5.2 GB" in out.getvalue()
    assert all(sp.command[0] != "lm_eval" for sp in docker.specs)


def test_ctrl_c_leaves_current_step_and_no_tmp(setup):
    s, docker, out = setup

    def interrupted(spec, log_path, on_line, on_tick=None):
        on_line(ev("step", n=1))
        on_line(ev("step", n=2))
        on_line(ev("step", n=3))
        raise KeyboardInterrupt

    s.run = interrupted
    with pytest.raises(KeyboardInterrupt):
        flow.run_quantize(s, quantize_run(s, ["int8-w8a8"]))
    assert s.current == "[3/4] Quantizing INT8-W8A8"
    assert not list(s.rd.rglob(".tmp-*"))
    assert docker.chowns and docker.chowns[-1][:3] == ["docker", "run", "--rm"]  # files handed back even on Ctrl-C


def test_run_eval_with_baseline(setup):
    s, docker, out = setup
    docker.evals["Qwen3.5-4B-quantized.w4a16+act-fp8"] = ("lm_eval/eval-rh-fp8.log", "lm_eval/out/rh-fp8", 0)
    docker.evals["baseline"] = ("lm_eval/eval-bf16-chat.log", "lm_eval/out/bf16-chat", 0)
    label = flow.label_for("RedHatAI/Qwen3.5-4B-quantized.w4a16", "fp8", None)
    r = flow.EvalRun("RedHatAI/Qwen3.5-4B-quantized.w4a16", label, ["gsm8k"], 50, "chat",
                     {"VLLM_MARLIN_INPUT_DTYPE": "fp8"}, None, [("quantization", "compressed-tensors · INT4 W4A16")],
                     5_000_000_000, "Qwen/Qwen3.5-4B", 9_078_531_072, {})
    assert flow.run_eval(s, r) == 0
    text = out.getvalue()
    assert "══════ [4/4] Evaluating baseline Qwen/Qwen3.5-4B " in text
    assert "  vs baseline  : gsm8k strict-match 0.820 → 0.780 · −4.0 pp · recovery 95.1%" in text
    spec = [sp for sp in docker.specs if sp.env.get("VLLM_MARLIN_INPUT_DTYPE")][0]
    assert "pretrained=RedHatAI/Qwen3.5-4B-quantized.w4a16," in arg(spec.command, "--model_args")


def test_quantization_text():
    assert flow.quantization_text(W8A8_QC) == "compressed-tensors · INT8 W8A8 symmetric"
    assert flow.quantization_text(None) == "none (BF16)"


def test_label_for():
    assert flow.label_for("/x/runs/m/checkpoints/INT4-W4A16", "int8", None) == "INT4-W4A16+act-int8"
    assert flow.label_for("Qwen/Qwen3.5-4B", None, "fp8") == "Qwen3.5-4B+fp8-online"


def test_ctrl_c_during_eval_removes_eval_tmp(setup):
    s, docker, out = setup

    def interrupted_eval(spec, log_path, on_line, on_tick=None):
        if spec.command[0] != "lm_eval":
            return docker(spec, log_path, on_line, on_tick)
        on_line("starting")
        raise KeyboardInterrupt

    s.run = interrupted_eval
    with pytest.raises(KeyboardInterrupt):
        flow.run_quantize(s, quantize_run(s, ["int4-w4a16"]))
    assert s.current == "[4/4] Evaluating bf16 baseline"
    assert not list(s.rd.rglob(".tmp-*"))


@pytest.mark.parametrize("on_error", ["continue", "stop"])
def test_failed_baseline_counts_as_failure(setup, on_error):
    s, docker, out = setup
    docker.evals["bf16"] = ("lm_eval/eval-bf16-docs.err-mamba.log", None, 1)
    code = flow.run_quantize(s, quantize_run(s, ["int4-w4a16"], on_error=on_error))
    assert code == 1
    evals = [sp for sp in docker.specs if sp.command[0] == "lm_eval"]
    assert len(evals) == (1 if on_error == "stop" else 2)


def test_quantize_without_saved_event_is_a_failure(setup):
    s, docker, out = setup
    docker.quantize["INT4-W4A16"] = "no-saved"
    code = flow.run_quantize(s, quantize_run(s, ["int4-w4a16", "int4-w4a8-fp8", "int8-w8a8"]))
    text = out.getvalue()
    assert code == 1
    assert "quantize finished without a saved event" in text
    assert not (s.rd / "checkpoints" / "INT4-W4A16").exists()
    assert not list(s.rd.rglob(".tmp-*"))
    assert "  int4-w4a8-fp8   SKIPPED" in text


def test_stale_state_is_refused(setup):
    s, docker, out = setup
    r = quantize_run(s, ["int4-w4a16"])
    r.states[0].action = "stale"
    with pytest.raises(ValueError, match="checkpoint INT4-W4A16 is stale"):
        flow.run_quantize(s, r)


def test_unreadable_results_json_fails_the_scheme(setup, tmp_path):
    s, docker, out = setup
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "results_x.json").write_text("{not json")
    docker.evals["int8-w8a8"] = ("lm_eval/eval-w8a8.log", bad, 0)
    assert flow.run_quantize(s, quantize_run(s, ["int8-w8a8"])) == 1
    text = out.getvalue()
    assert "unreadable lm_eval results: JSONDecodeError" in text
    assert "  int8-w8a8       FAILED" in text
    assert not list(s.rd.rglob(".tmp-*"))


def test_exit_137_without_memory_limit_does_not_claim_a_limit():
    # probe and eval containers run without --memory: exit 137 there is the host OOM killer or an external kill
    assert flow.failure_reason(137, [], None) == "container killed (exit 137): host out of memory or killed externally"
    assert flow.failure_reason(137, [], 25 * 10**9) == (
        "container killed (exit 137): host RAM ran out at the --memory limit of 25.0 GB")
