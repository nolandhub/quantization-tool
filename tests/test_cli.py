# ABOUTME: Tests for qtool.cli: argument validation (exit 2), dry-run, preflight blocking before any container,
# ABOUTME: a full fake quantize run, stale checkpoint, Ctrl-C (exit 130) and `qtool eval --act` checks.
import collections
import dataclasses
import io
import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from fakes import W4A16_QC, W8A8_QC, FakeDocker
from qtool import __version__, cli, preflight
from qtool.console import Console

ROOT = Path(__file__).resolve().parents[1]
GIB_KB = 1024 * 1024


class FakeSystem(cli.System):
    def __init__(self, fixtures, tmp_path, gpu_csv="rtx4070", ram_gb=27):
        self.fixtures, self.tmp = fixtures, tmp_path
        self.lock_path = tmp_path / ".lock"
        self.gpu_csv, self.ram_gb = gpu_csv, ram_gb
        self.built = []
        self.chowns = []
        self.docker = None
        self.gpu_err = None
        self.sm = None
        ticks = itertools.count(0, 5)
        self.clock = lambda: next(ticks)

    def run(self, spec, log_path, on_line, on_tick=None):
        if self.docker is None:
            self.docker = FakeDocker(self.fixtures, Path(spec.mounts[3][0]))
            self.docker.evals = collections.defaultdict(lambda: ("lm_eval/eval-bf16-chat.log", "lm_eval/out/bf16-chat", 0), {"bf16": ("lm_eval/eval-bf16-chat.log", "lm_eval/out/bf16-chat", 0),
                                 "int4-w4a16": ("lm_eval/eval-w4a16-none.log", "lm_eval/out/w4a16-none", 0),
                                 "INT4-W4A16+act-int8": ("lm_eval/eval-w4a16-int8.log", "lm_eval/out/w4a16-int8", 0)})
            self.docker.probes = getattr(self, "probes", {})
        return self.docker(spec, log_path, on_line, on_tick)

    def chown(self, argv, **kw):
        self.chowns.append(argv)

    def docker_error(self):
        return None

    def gpu_error(self, image, index):
        return self.gpu_err

    def gpus(self):
        f = self.fixtures / "nvidia-smi"
        apps = f / ("apollo-8gpu-apps.csv" if self.gpu_csv.startswith("apollo") else "rtx4070-apps.csv")
        uuid = f / ("apollo-8gpu-uuid.csv" if self.gpu_csv.startswith("apollo") else "rtx4070-uuid.csv")
        gpus = preflight.parse_gpus((f / f"{self.gpu_csv}.csv").read_text())
        if self.sm:
            gpus = [dataclasses.replace(g, cc=self.sm) for g in gpus]
        return gpus, preflight.parse_apps(apps.read_text(), uuid.read_text())

    def container_names(self):
        return {}

    def meminfo(self):
        return f"MemAvailable: {self.ram_gb * 10 ** 9 // 1024} kB\n"

    def disk_free(self, path):
        return 659 * 10 ** 9

    def user(self):
        return "1002:1002"

    def ensure_image(self, kind, cuda, console):
        self.built.append((kind, cuda))
        return f"qtool-{kind}:{cuda or 'x'}-fake"


@pytest.fixture
def env(fixtures, tmp_path):
    out = io.StringIO()
    system = FakeSystem(fixtures, tmp_path)

    def run(*argv):
        code = cli.main([*argv, "--output-dir", str(tmp_path / "runs"), "--hf-cache", str(tmp_path / "hf")], system,
                        Console(out, tty=False, wallclock=lambda: "21:00:00", clock=lambda: 0.0))
        return code, out.getvalue()

    return system, run


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"qtool {__version__}"


def test_bin_shim():
    r = subprocess.run([sys.executable, str(ROOT / "bin" / "qtool"), "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == f"qtool {__version__}"


@pytest.mark.parametrize("argv,err", [
    (["quantize", "m"], "required: --scheme"),
    (["quantize", "m", "--scheme", "int4-w4a16", "--eval-limit", "0"], "expected a positive integer, got '0'"),
    (["eval", "m", "--act", "fp8", "--quantization", "fp8"], "not allowed with argument"),
    (["eval", "m", "--act", "int4"], "invalid choice"),
])
def test_argparse_errors_exit_2(capsys, argv, err):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2 and err in capsys.readouterr().err


def test_unknown_scheme_is_usage_error(env):
    system, run = env
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a8")
    assert code == 2 and "error: unknown scheme 'int4-w4a8'" in text and system.docker is None


def test_dry_run_prints_plan_and_runs_nothing(env):
    system, run = env
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16,int4-w4a8-fp8", "--dry-run")
    assert code == 0
    assert "  quantize  INT4-W4A16  -> shared by: int4-w4a16, int4-w4a8-fp8" in text
    assert system.docker is None and system.built == [] and not system.lock_path.exists()


def test_multi_gpu_needs_flag(fixtures, tmp_path):
    out = io.StringIO()
    code = cli.main(["quantize", "m", "--scheme", "int4-w4a16", "--no-eval", "--output-dir", str(tmp_path)],
                    FakeSystem(fixtures, tmp_path, "apollo-8gpu"), Console(out, tty=False))
    assert code == 2 and "this machine has 8 GPUs: choose one with --gpu N" in out.getvalue()


def test_full_quantize_run(env):
    system, run = env
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--eval-limit", "50")
    assert code == 0, text
    assert "[preflight] GPU 0 · SM 8.9 · VRAM free 15.6 / 16.0 GiB (need 5.4 GB) · RAM avail 27.0 GB (need 14.5 GB)" in text
    assert "[preflight] eval mode chat (chat template keeps message content)" in text
    assert "  eval      bf16 baseline (cache: none, will run)" in text
    assert "Summary · gsm8k strict-match (limit 50)" in text
    assert system.built == [("compress", "cu129"), ("eval", None)]
    assert len(system.chowns) == len(system.docker.specs)  # every container is followed by a chown
    assert system.chowns[0][-4:] == ["-R", "1002:1002", "/runs", "/qtool-home"]
    assert not system.lock_path.exists()


def test_ram_threshold_blocks_before_any_quantize(env):
    system, run = env
    system.ram_gb = 8
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16")
    assert code == 2
    assert "[preflight] ✗ host has 8.0 GB RAM available, quantize needs 14.5 GB" in text
    assert [sp.command[1] for sp in system.docker.specs] == ["/qtool/probe.py"]


def test_skip_preflight_ignores_thresholds(env):
    system, run = env
    system.ram_gb = 8
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--no-eval", "--skip-preflight")
    assert code == 0, text


def test_already_quantized_model_is_refused(env):
    system, run = env
    system.probes = {"RedHatAI/x": {"quantized": True, "quantization_config": W4A16_QC}}
    code, text = run("quantize", "RedHatAI/x", "--scheme", "int4-w4a16")
    assert code == 2 and "[preflight] ✗ RedHatAI/x is already quantized (compressed-tensors): evaluate it with `qtool eval`" in text


def test_unknown_task_is_refused(env):
    system, run = env
    system.probes = {"Qwen/Qwen3.5-4B": {"tasks_missing": ["gsm9k"]}}
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--eval", "gsm9k")
    assert code == 2 and "unknown lm_eval task 'gsm9k'" in text


def test_stale_checkpoint_needs_force(env):
    system, run = env
    assert run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--no-eval")[0] == 0
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--no-eval", "--calib-samples", "256")
    assert code == 2
    assert "error: checkpoints/INT4-W4A16 was made with different settings: calib_samples: 512 → 256" in text
    assert "re-run with --force to rebuild it" in text
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--no-eval", "--calib-samples", "256", "--force")
    assert code == 0


def test_ctrl_c_exits_130_and_releases_lock(env):
    system, run = env

    def interrupted(spec, log_path, on_line, on_tick=None):
        if "/qtool/probe.py" in spec.command:
            return FakeSystem.run(system, spec, log_path, on_line, on_tick)
        raise KeyboardInterrupt

    system.run = interrupted
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--no-eval")
    assert code == 130 and "Interrupted during [1/4] Loading model · INT4-W4A16" in text
    assert not system.lock_path.exists()


def test_eval_act_on_qtool_checkpoint(env, tmp_path):
    system, run = env
    run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--no-eval")
    ckpt = tmp_path / "runs" / "Qwen3.5-4B" / "checkpoints" / "INT4-W4A16"
    system.docker.probes = {str(ckpt): {"quantized": True, "quantization_config": W4A16_QC, "weight_bytes": 5_200_000_000}}
    code, text = run("eval", str(ckpt), "--act", "int8", "--eval-limit", "50")
    assert code == 0, text
    assert "  runtime env  : VLLM_MARLIN_INPUT_DTYPE=int8" in text
    spec = [sp for sp in system.docker.specs if sp.command[0] == "lm_eval"][-1]
    assert spec.env["VLLM_MARLIN_INPUT_DTYPE"] == "int8" and "pretrained=/runs/checkpoints/INT4-W4A16," in spec.command[4]


def test_eval_act_rejects_w8a8(env):
    system, run = env
    system.probes = {"m/w8a8": {"quantized": True, "quantization_config": W8A8_QC}}
    code, text = run("eval", "m/w8a8", "--act", "fp8")
    assert code == 2 and "--act needs an INT4 symmetric group-wise checkpoint" in text


def test_missing_nvidia_smi_is_preflight_error(fixtures, tmp_path, monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(preflight, "query_gpus", missing)
    system = cli.System()
    with pytest.raises(preflight.PreflightError, match="nvidia-smi not found"):
        system.gpus()
    out = io.StringIO()
    system.docker_error = lambda: None
    code = cli.main(["quantize", "m", "--scheme", "int4-w4a16", "--output-dir", str(tmp_path)], system, Console(out, tty=False))
    assert code == 2 and "error: nvidia-smi not found" in out.getvalue()


def test_gpu_not_visible_in_container_blocks(env):
    system, run = env
    system.gpu_err = "boom"
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16")
    assert code == 2 and "error: boom" in text
    assert system.docker is None and not system.lock_path.exists()
    code, text = run("eval", "Qwen/Qwen3.5-4B")
    assert code == 2 and "error: boom" in text and system.docker is None and not system.lock_path.exists()


def test_gpu_error_command(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, "", "no devices\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    msg = cli.System().gpu_error("img:1", 3)
    assert msg == "containers cannot see GPU 3: no devices"
    assert calls[0] == ["docker", "run", "--rm", "--gpus", "device=3", "--entrypoint", "nvidia-smi", "img:1", "-L"]


def test_sm70_quantize_with_eval_blocks_before_any_build(fixtures, tmp_path):
    out = io.StringIO()
    system = FakeSystem(fixtures, tmp_path, "apollo-8gpu")
    code = cli.main(["quantize", "m", "--scheme", "int4-w4a16", "--gpu", "1", "--output-dir", str(tmp_path / "runs")],
                    system, Console(out, tty=False))
    assert code == 2 and "eval cannot run on SM 7.0" in out.getvalue()
    assert system.built == [] and system.docker is None and not system.lock_path.exists()


def test_stale_checkpoint_refused_before_any_build(env):
    system, run = env
    assert run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--no-eval")[0] == 0
    built, nspecs = list(system.built), len(system.docker.specs)
    code, text = run("quantize", "Qwen/Qwen3.5-4B", "--scheme", "int4-w4a16", "--no-eval", "--calib-samples", "256")
    assert code == 2 and "re-run with --force to rebuild it" in text
    assert system.built == built and len(system.docker.specs) == nspecs


def test_eval_act_dry_run_checks_sm(env):
    system, run = env
    system.sm = "8.0"
    code, text = run("eval", "m", "--act", "fp8", "--dry-run")
    assert code == 2 and "--act needs SM 8.9 or 12.0" in text
    assert system.docker is None and system.built == [] and not system.lock_path.exists()


def test_eval_baseline_completion_mode(env):
    system, run = env
    system.probes = {"m/base": {"chat_mode": "completion", "chat_reason": "template drops content"}}
    code, text = run("eval", "Qwen/Qwen3.5-4B", "--baseline", "m/base", "--eval-limit", "50")
    assert "[preflight] eval mode completion (template drops content)" in text
    cmds = [sp.command for sp in system.docker.specs if sp.command[0] == "lm_eval"]
    assert cmds and all("--apply_chat_template" not in c for c in cmds)


def test_eval_dry_run_prints_plan(env):
    system, run = env
    code, text = run("eval", "Qwen/Qwen3.5-4B", "--dry-run")
    assert code == 0 and "Plan for eval of Qwen/Qwen3.5-4B" in text
    assert system.docker is None and system.built == [] and not system.lock_path.exists()


def test_eval_bf16_uses_model_id_and_skips_when_too_big(env):
    system, run = env
    code, text = run("eval", "Qwen/Qwen3.5-4B", "--eval-limit", "50")
    assert code == 0, text
    spec = [sp for sp in system.docker.specs if sp.command[0] == "lm_eval"][-1]
    assert "pretrained=Qwen/Qwen3.5-4B," in spec.command[4]
    system.docker.probes = {"big/model": {"params": 10_000_000_000}}
    code, text = run("eval", "big/model", "--eval-limit", "50")
    assert "SKIPPED" in text


def test_quantized_baseline_uses_weight_bytes(env):
    system, run = env
    system.probes = {"m/q": {"quantized": True, "quantization_config": W4A16_QC, "weight_bytes": 5_200_000_000,
                             "params": 10_000_000_000}}
    code, text = run("eval", "Qwen/Qwen3.5-4B", "--baseline", "m/q", "--eval-limit", "50")
    assert "SKIPPED" not in text, text


def test_lock_contention_exits_2(env):
    system, run = env
    system.lock_path.write_text("1")
    code, text = run("eval", "Qwen/Qwen3.5-4B")
    assert code == 2 and "another qtool run (pid 1) is active" in text


def test_docker_error_exits_2(env):
    system, run = env
    system.docker_error = lambda: "docker is not installed"
    code, text = run("eval", "Qwen/Qwen3.5-4B")
    assert code == 2 and "error: docker is not installed" in text


def test_ctrl_c_before_lock_exits_130(env):
    system, run = env

    def boom():
        raise KeyboardInterrupt

    system.docker_error = boom
    code, text = run("eval", "Qwen/Qwen3.5-4B")
    assert code == 130 and "Interrupted" in text


def test_multi_gpu_table_labels_use_container_names(fixtures, tmp_path):
    out = io.StringIO()
    system = FakeSystem(fixtures, tmp_path, "apollo-8gpu")
    system.container_names = lambda: {i: "named-ctr" for i in ("x",)}
    called = []
    orig = preflight.process_label
    preflight.process_label = lambda a, names: (called.append(a), orig(a, names))[1]
    try:
        cli.main(["eval", "m", "--output-dir", str(tmp_path)], system, Console(out, tty=False))
    finally:
        preflight.process_label = orig
    assert called
