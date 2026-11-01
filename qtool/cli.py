# ABOUTME: Command-line interface: parsing and validation for `qtool quantize` / `qtool eval`, preflight wiring
# ABOUTME: (GPU, images, probe, thresholds) and exit codes: 0 ok, 1 a scheme failed, 2 usage/preflight error, 130 Ctrl-C.
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from qtool import __version__, flow, images, plan, preflight, schemes
from qtool.console import Console, fmt_gb
from qtool.runner import run_container

DEFAULT_CALIB = "HuggingFaceH4/ultrachat_200k:train_sft"
GB = 10 ** 9


class UsageError(Exception):
    pass


def positive_int(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got '{text}'") from None
    if n <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got '{text}'")
    return n


def eval_limit(text: str) -> int | None:
    return None if text == "full" else positive_int(text)


def _eval_args(p, quantize: bool) -> None:
    g = p.add_argument_group("evaluation (step 4)")
    g.add_argument("--eval", default="gsm8k", metavar="TASKS", help="lm_eval tasks, comma separated [default: gsm8k]")
    g.add_argument("--eval-limit", type=eval_limit, default=None, metavar="N|full", help="questions per task [default: full]")
    if quantize:
        g.add_argument("--no-eval", action="store_true", help="skip step 4")


def _common_args(p, quantize: bool) -> None:
    g = p.add_argument_group("run control")
    g.add_argument("--output-dir", default="~/qtool/runs", metavar="DIR", help="runs directory [default: ~/qtool/runs]")
    if quantize:
        g.add_argument("--on-error", choices=["continue", "stop"], default="continue", help="[default: continue]")
        g.add_argument("--force", action="store_true", help="quantize again even if the checkpoint exists")
    g.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    g.add_argument("--skip-preflight", action="store_true", help="skip VRAM / RAM threshold checks (at your own risk)")
    h = p.add_argument_group("hardware")
    h.add_argument("--gpu", type=int, metavar="N", help="GPU index (nvidia-smi); required with 2+ GPUs")
    if quantize:
        h.add_argument("--compute-dtype", choices=["auto", "bfloat16", "float32"], default="auto",
                       help="auto: SM >= 8.0 bfloat16, else float32 (never float16) [default: auto]")
    h.add_argument("--hf-cache", default=os.environ.get("HF_HOME") or "~/.cache/huggingface", metavar="DIR",
                   help="HF cache mounted into the containers [default: $HF_HOME, else ~/.cache/huggingface]")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qtool", description="Quantize a model with LLM Compressor and evaluate it in vLLM.")
    p.add_argument("--version", action="version", version=f"qtool {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("quantize", help="steps 1-4: load, calibrate, quantize into the requested schemes, evaluate")
    q.add_argument("model", metavar="MODEL", help="HF repo id or local dir of the unquantized model")
    q.add_argument("--scheme", required=True, metavar="SCHEMES", help=f"comma separated: {', '.join(schemes.SCHEMES)}")
    g = q.add_argument_group("calibration (step 2)")
    g.add_argument("--calib", default=DEFAULT_CALIB, metavar="SOURCE",
                   help=f"HF dataset id[:split] or .jsonl with 'messages' or 'text' [default: {DEFAULT_CALIB}]")
    g.add_argument("--calib-samples", type=positive_int, default=512, metavar="N", help="[default: 512]")
    g.add_argument("--calib-seq-len", type=positive_int, default=2048, metavar="N", help="[default: 2048]")
    r = q.add_argument_group("recipe (step 3)")
    r.add_argument("--ignore", default="", metavar="PATTERNS",
                   help="extra ignore patterns (re:... or module names), comma separated; added to the derived list")
    _eval_args(q, True)
    _common_args(q, True)
    e = sub.add_parser("eval", help="step 4 only: evaluate any model vLLM can load")
    e.add_argument("model", metavar="MODEL", help="qtool checkpoint, quantized HF model or BF16 model")
    x = e.add_mutually_exclusive_group()
    x.add_argument("--act", choices=["fp8", "int8"], help="W4A8 via Marlin (INT4 symmetric group-wise checkpoint)")
    x.add_argument("--quantization", choices=["fp8"], help="vLLM online FP8 quantization")
    e.add_argument("--baseline", metavar="MODEL", help="also evaluate this model (cached) for Δ pp and recovery %%")
    _eval_args(e, False)
    _common_args(e, False)
    return p


class System:
    """Host side effects (docker, nvidia-smi, /proc); tests swap in a fake."""

    lock_path = Path("~/qtool/.lock").expanduser()
    run = staticmethod(run_container)
    chown = staticmethod(subprocess.run)
    clock = staticmethod(time.monotonic)

    def docker_error(self) -> str | None:
        try:
            r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True)
        except FileNotFoundError:
            return "docker is not installed"
        return None if r.returncode == 0 else f"docker is not usable: {(r.stderr or r.stdout).strip()}"

    def gpus(self):
        try:
            return preflight.query_gpus()
        except FileNotFoundError:
            raise preflight.PreflightError("nvidia-smi not found: no NVIDIA driver on this host") from None

    def gpu_error(self, image: str, index: int) -> str | None:
        r = subprocess.run(["docker", "run", "--rm", "--gpus", f"device={index}", "--entrypoint", "nvidia-smi", image, "-L"],
                           capture_output=True, text=True)
        return None if r.returncode == 0 else f"containers cannot see GPU {index}: {(r.stderr or r.stdout).strip()}"

    def container_names(self) -> dict[str, str]:
        r = subprocess.run(["docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"], capture_output=True, text=True)
        return dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)

    def meminfo(self) -> str:
        return Path("/proc/meminfo").read_text()

    def disk_free(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def user(self) -> str:
        return f"{os.getuid()}:{os.getgid()}"

    def ensure_image(self, kind: str, cuda: str | None, console: Console) -> str:
        tag = images.tag(kind, cuda)
        if not images.exists(tag):
            console.line(f"[setup] building {tag} (first run only, takes several minutes)")
            r = subprocess.run(images.build_argv(kind, cuda))
            if r.returncode != 0:
                raise preflight.PreflightError(f"docker build of {tag} failed (exit {r.returncode})")
        return tag


def _tasks(text: str) -> list[str]:
    tasks = [t.strip() for t in text.split(",") if t.strip()]
    if not tasks:
        raise UsageError("--eval is empty")
    return tasks


def _gpu(args, system: System, console: Console) -> preflight.Gpu:
    if err := system.docker_error():
        raise preflight.PreflightError(err)
    gpus, apps = system.gpus()
    names = system.container_names()
    gpu = preflight.select_gpu(gpus, args.gpu, apps, lambda a: preflight.process_label(a, names))
    console.line(f"qtool {__version__} · {time.strftime('%Y-%m-%d %H:%M:%S')} · {gpu.name} "
                 f"{gpu.total_mib / 1024:.0f} GiB SM {gpu.cc}")
    busy = [a for a in apps if a.gpu_index == gpu.index]
    if busy:
        console.line("[preflight] in use on this GPU: " + ", ".join(
            f"{preflight.process_label(a, names)} {a.used_mib} MiB" for a in busy))
    return gpu


def probe_findings(info: dict, quantize: bool, no_eval: bool = False) -> list[preflight.Finding]:
    out = []
    arch = (info.get("architectures") or ["unknown"])[0]
    if quantize and info["quantized"]:
        method = (info.get("quantization_config") or {}).get("quant_method", "unknown")
        out.append(preflight.Finding("error", f"{info['model']} is already quantized ({method}): evaluate it with `qtool eval`"))
    if quantize and info["remote_code"]:
        out.append(preflight.Finding("error", f"{info['model']} needs trust_remote_code, which qtool does not support"))
    if quantize and arch not in flow.VERIFIED_ARCHS:
        out.append(preflight.Finding("warn", f"{arch} [unverified architecture]: the recipe is derived from config + modules; "
                                             "check the ignore list at [3/4]" + (" and run eval later" if no_eval else "")))
    out += [preflight.Finding("error", f"unknown lm_eval task '{t}'") for t in info.get("tasks_missing", [])]
    return out


def _report(console: Console, findings: list[preflight.Finding]) -> bool:
    for f in findings:
        console.line(f"[preflight] {'✗' if f.level == 'error' else '!'} {f.message}")
    return any(f.level == "error" for f in findings)


def _new(findings: list[preflight.Finding], seen: list[preflight.Finding]) -> list[preflight.Finding]:
    """Findings not already reported (the GPU-only checks run before the probe, then again with it)."""
    return [f for f in findings if f not in seen]


def _ensure_gpu_visible(system: System, image: str, gpu: preflight.Gpu) -> None:
    if err := system.gpu_error(image, gpu.index):
        raise preflight.PreflightError(err)


def cmd_quantize(args, system: System, console: Console) -> int:
    try:
        names = schemes.parse_schemes(args.scheme)
    except ValueError as e:
        raise UsageError(str(e)) from None
    tasks = _tasks(args.eval)
    if args.calib.endswith(".jsonl") and not Path(args.calib).expanduser().exists():
        raise UsageError(f"--calib file not found: {args.calib}")
    output = Path(args.output_dir).expanduser()
    rd = plan.runs_dir(output, args.model)
    gpu = _gpu(args, system, console)
    dtype = schemes.compute_dtype(gpu.cc, args.compute_dtype)
    ignore = [p.strip() for p in args.ignore.split(",") if p.strip()]
    needed = schemes.checkpoints_for(names)
    llmc = images.pinned("compress", "llmcompressor")
    sigs = {c: plan.checkpoint_signature(args.model, c, args.calib, args.calib_samples, args.calib_seq_len, ignore, llmc)
            for c in needed}
    states = plan.checkpoint_states(rd, needed, sigs, args.force)
    early = preflight.check_quantize(gpu, None, dtype, 0, names, args.no_eval, thresholds=False)
    blocked = _report(console, early)
    if args.dry_run:
        for line in plan.render_plan(args.model, states, names, tasks, args.eval_limit, "cache not checked in --dry-run",
                                     args.no_eval):
            console.line(line)
    if stale := [st for st in states if st.action == "stale"]:
        for st in stale:
            console.line(f"error: checkpoints/{st.name} was made with different settings: " + "; ".join(st.diff))
        console.line("re-run with --force to rebuild it")
    if blocked or stale:
        return 2
    if args.dry_run:
        return 0
    lock = preflight.RunLock(system.lock_path)
    lock.acquire()
    s = None
    try:
        for d in (rd / "logs", output / ".home"):
            d.mkdir(parents=True, exist_ok=True)
        compress = system.ensure_image("compress", images.compress_cuda(gpu.cc), console)
        evali = None if args.no_eval else system.ensure_image("eval", None, console)
        _ensure_gpu_visible(system, compress, gpu)
        s = flow.Session(console, gpu, rd, Path(args.hf_cache).expanduser(), output / ".home", system.user(), compress,
                         evali, run=system.run, chown=system.chown, clock=system.clock)
        s.started = s.clock()
        info = flow.probe(s, evali or compress, args.model, [] if args.no_eval else tasks)
        ram = preflight.mem_available_bytes(system.meminfo())
        findings = probe_findings(info, True, args.no_eval) + _new(preflight.check_quantize(
            gpu, info["params"], dtype, ram, names, args.no_eval, thresholds=not args.skip_preflight), early)
        if not args.no_eval and not args.skip_preflight:
            findings += preflight.check_eval_free(gpu)
        need = preflight.quantize_needs(info["params"], dtype) if info["params"] else None
        console.line(f"[preflight] GPU {gpu.index} · SM {gpu.cc} · VRAM free {gpu.free_mib / 1024:.1f} / "
                     f"{gpu.total_mib / 1024:.1f} GiB" + (f" (need {fmt_gb(need[0])})" if need else "")
                     + f" · RAM avail {fmt_gb(ram)}" + (f" (need {fmt_gb(need[1])})" if need else ""))
        console.line(f"[preflight] compute dtype {dtype} · image {compress} · disk free {fmt_gb(system.disk_free(output))}")
        if not args.no_eval:
            console.line(f"[preflight] eval mode {info['chat_mode']} ({info['chat_reason']})")
        if _report(console, findings):
            return 2
        s.memory_bytes = max(ram - 2 * GB, GB)
        note = None
        if not args.no_eval:
            meta = plan.eval_meta(args.model, tasks[0], args.eval_limit, evali, info["chat_mode"], gpu.name)
            note = plan.cached_result(plan.result_path(rd, "bf16", tasks[0]), meta)[1]
        console.line("")
        for line in plan.render_plan(args.model, states, names, tasks, args.eval_limit, note, args.no_eval):
            console.line(line)
        options = flow.QuantizeOptions(args.model, args.calib, args.calib_samples, args.calib_seq_len, ignore, dtype,
                                       info["params"] * 2 if info["params"] else None)
        run = flow.QuantizeRun(args.model, names, states, sigs, options, info["params"], tasks, args.eval_limit,
                               info["chat_mode"], args.no_eval, args.on_error,
                               {"qtool": __version__, "model": args.model, "gpu": gpu.name, "tasks": tasks,
                                "limit": args.eval_limit, "mode": info["chat_mode"], "eval_image": evali,
                                "compress_image": compress, "compute_dtype": dtype})
        return flow.run_quantize(s, run)
    except KeyboardInterrupt:
        console.line(f"\nInterrupted during {s.current if s and s.current else 'preflight'}")
        return 130
    finally:
        lock.release()


def cmd_eval(args, system: System, console: Console) -> int:
    tasks = _tasks(args.eval)
    output = Path(args.output_dir).expanduser()
    rd = plan.runs_dir(output, args.model)
    gpu = _gpu(args, system, console)
    label = flow.label_for(args.model, args.act, args.quantization)
    rows = ["int4-w4a8-" + args.act] if args.act else (["fp8-w8a8"] if args.quantization else [])
    early = preflight.check_eval_sm(gpu, rows, baseline=bool(args.baseline))
    if args.act:
        early += [f for f in preflight.check_act(gpu, None) if f.message.startswith("--act needs SM")]
    blocked = _report(console, early)
    if args.dry_run:
        console.line(f"Plan for eval of {args.model}")
        if args.baseline:
            console.line(f"  eval      baseline {args.baseline} (cache not checked in --dry-run)")
        console.line(f"  eval      {label}   tasks: {','.join(tasks)} ({args.eval_limit or 'full'})")
    if blocked:
        return 2
    if args.dry_run:
        return 0
    lock = preflight.RunLock(system.lock_path)
    lock.acquire()
    s = None
    try:
        for d in (rd / "logs", output / ".home"):
            d.mkdir(parents=True, exist_ok=True)
        evali = system.ensure_image("eval", None, console)
        _ensure_gpu_visible(system, evali, gpu)
        s = flow.Session(console, gpu, rd, Path(args.hf_cache).expanduser(), output / ".home", system.user(), None, evali,
                         run=system.run, chown=system.chown, clock=system.clock)
        s.started = s.clock()
        info = flow.probe(s, evali, args.model, tasks)
        binfo = flow.probe(s, evali, args.baseline) if args.baseline else None
        findings = probe_findings(info, False)
        if args.act:
            findings += _new(preflight.check_act(gpu, info["quantization_config"]), early)
        if not args.skip_preflight:
            findings += preflight.check_eval_free(gpu)
        mode = "chat" if info["chat_mode"] == "chat" and (binfo is None or binfo["chat_mode"] == "chat") else "completion"
        reason = info["chat_reason"] if info["chat_mode"] != "chat" or binfo is None else binfo["chat_reason"]
        console.line(f"[preflight] GPU {gpu.index} · SM {gpu.cc} · VRAM free {gpu.free_mib / 1024:.1f} / "
                     f"{gpu.total_mib / 1024:.1f} GiB · image {evali}")
        console.line(f"[preflight] eval mode {mode} ({reason})")
        if _report(console, findings):
            return 2
        env = {"VLLM_MARLIN_INPUT_DTYPE": args.act} if args.act else {}
        params = info["params"]
        weight = info["weight_bytes"] if info["quantized"] else \
            (None if params is None else params if args.quantization else params * 2)
        qtext = flow.quantization_text(info["quantization_config"]) if info["quantized"] else \
            (f"vLLM online {args.quantization}" if args.quantization else "none (BF16)")
        rowinfo = [("model", args.model), ("quantization", qtext)]
        if env:
            rowinfo.append(("runtime env", " ".join(f"{k}={v}" for k, v in env.items())))
        bbytes = None
        if binfo:
            bbytes = binfo["weight_bytes"] if binfo["quantized"] else (binfo["params"] * 2 if binfo["params"] else None)
        run = flow.EvalRun(args.model, label, tasks, args.eval_limit, mode, env, args.quantization, rowinfo, weight,
                           args.baseline, bbytes, {"qtool": __version__, "model": args.model, "baseline": args.baseline,
                                                   "gpu": gpu.name, "tasks": tasks, "limit": args.eval_limit,
                                                   "mode": mode, "eval_image": evali})
        return flow.run_eval(s, run)
    except KeyboardInterrupt:
        console.line(f"\nInterrupted during {s.current if s and s.current else 'preflight'}")
        return 130
    finally:
        lock.release()


def main(argv=None, system: System | None = None, console: Console | None = None) -> int:
    args = build_parser().parse_args(argv)
    system, console = system or System(), console or Console()
    try:
        return (cmd_quantize if args.command == "quantize" else cmd_eval)(args, system, console)
    except KeyboardInterrupt:
        console.line("\nInterrupted")
        return 130
    except (UsageError, preflight.PreflightError) as e:
        console.line(f"error: {e}")
        return 2
