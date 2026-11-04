# ABOUTME: Orchestrates runs: step headers, rendering container events and progress, quantize and eval jobs,
# ABOUTME: --on-error, the summary table and exit codes. Containers start through an injected runner (fake in tests).
import itertools
import json
import os
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from qtool import images, parsers, plan, preflight, report, schemes
from qtool.console import Console, fmt_duration, fmt_gb
from qtool.runner import ContainerSpec, atomic_dir, run_container

VERIFIED_ARCHS = {"Qwen3_5ForConditionalGeneration", "Phi3ForCausalLM", "OlmoeForCausalLM",
                  "SmolVLMForConditionalGeneration"}
TEMPLATES = {"chat_template(str)": "chat template (string content)",
             "chat_template(parts)": "chat template (list-of-parts content)",
             "raw": "raw text (chat template missing or drops message content)",
             "text": "text column (no template)"}
EVAL_ARGS = "max_model_len=4096,gpu_memory_utilization=0.85,max_num_seqs=64,enable_thinking=False"


class StepFailed(Exception):
    def __init__(self, reason: str, step: int | None = None):
        super().__init__(reason)
        self.reason, self.step = reason, step


@dataclass
class Session:
    console: Console
    gpu: preflight.Gpu
    rd: Path  # runs/<model>
    hf_cache: Path
    home: Path
    user: str
    compress_image: str | None = None
    eval_image: str | None = None
    memory_bytes: int | None = None
    run: object = run_container
    chown: object = subprocess.run
    clock: object = time.monotonic
    current: str = ""
    started: float = 0.0
    _seq: itertools.count = field(default_factory=itertools.count)

    def header(self, title: str) -> None:
        self.current = title
        self.console.header(title)

    def name(self) -> str:
        return f"qtool-{os.getpid()}-{next(self._seq)}"

    def run_container(self, spec: ContainerSpec, log_path: Path, on_line, on_tick=None) -> int:
        """Containers run as root (the HF cache is usually root-owned); hand files back to the developer afterwards."""
        try:
            return self.run(spec, log_path, on_line, on_tick)
        finally:
            self.chown(["docker", "run", "--rm", "-v", f"{self.rd}:/runs", "-v", f"{self.home}:/qtool-home",
                        "--entrypoint", "chown", spec.image, "-R", self.user, "/runs", "/qtool-home"], capture_output=True)


def mount_path(path_or_id: str, ctr: str) -> tuple[str, list]:
    """Local files/dirs are mounted read-only at ctr; HF ids pass through unchanged."""
    p = Path(path_or_id).expanduser()
    if p.exists():
        return ctr, [(str(p.resolve()), ctr, True)]
    return path_or_id, []


def container_model(s: Session, model: str) -> tuple[str, list]:
    p = Path(model).expanduser()
    if p.exists() and p.resolve().is_relative_to(s.rd.resolve()):
        return "/runs/" + str(p.resolve().relative_to(s.rd.resolve())), []
    return mount_path(model, "/model")


def container_spec(s: Session, image: str, command: list[str], mounts=(), env=None, gpu=True, memory=False) -> ContainerSpec:
    e = {"HF_HOME": "/hf", "HOME": "/qtool-home", "PYTHONUNBUFFERED": "1", **(env or {})}
    if os.environ.get("HF_TOKEN"):
        e["HF_TOKEN"] = None
    base = [(str(s.hf_cache), "/hf", False), (str(s.home), "/qtool-home", False), (str(images.SCRIPTS), "/qtool", True),
            (str(s.rd), "/runs", False)]
    return ContainerSpec(image=image, command=command, name=s.name(), gpu=s.gpu.index if gpu else None, env=e,
                         mounts=[*base, *mounts], memory_bytes=s.memory_bytes if memory else None)


def failure_reason(code: int, lines, memory_bytes: int | None) -> str:
    err = parsers.find_error(list(lines))
    if code == 137 and err is None:
        if memory_bytes is None:  # no --memory on this container: the host OOM killer or an external kill
            return "container killed (exit 137): host out of memory or killed externally"
        return f"container killed (exit 137): host RAM ran out at the --memory limit of {fmt_gb(memory_bytes)}"
    if err:
        return err[0] + (f" ({err[1]})" if err[1] else "")
    return f"container exited with code {code}"


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


# ---- probe ----------------------------------------------------------------------------------------------------
def probe(s: Session, image: str, model: str, tasks=()) -> dict:
    m, mounts = mount_path(model, "/model")
    found, tail = {}, deque(maxlen=200)

    def on_line(line):
        tail.append(line)
        if e := parsers.parse_event(line):
            found.update(e)

    cmd = ["python3", "/qtool/probe.py", "--model", m, *(["--tasks", ",".join(tasks)] if tasks else [])]
    code = s.run_container(container_spec(s, image, cmd, mounts, gpu=False), s.rd / "logs" / "probe.log", on_line)
    if found.get("event") == "probe":
        return found
    if found.get("event") == "probe_error":
        raise preflight.PreflightError(f"cannot read model {model}: {found['message']}")
    raise preflight.PreflightError(f"probe container failed: {failure_reason(code, tail, None)}")


# ---- quantize (steps 1-3) -----------------------------------------------------------------------------------
@dataclass
class QuantizeOptions:
    model: str
    calib: str
    samples: int
    seq_len: int
    ignore: list[str]
    compute_dtype: str
    source_bytes: int | None


def _short(pattern: str) -> str:
    return pattern.removeprefix("re:").removeprefix(".*").removesuffix(".*").removesuffix("$").replace("\\", "")


def _show_recipe(c: Console, ckpt: str, r: dict | None) -> None:
    if not r:
        return
    sequential = "" if schemes.data_free(ckpt) else f" · sequential {r['sequential_target']}"
    c.kv("recipe", schemes.CHECKPOINTS[ckpt]["recipe"] + sequential)
    c.kv("ignore", " · ".join(f"{_short(p)} ({why})" for p, why in r["why"].items()))
    groups = sorted({x.split(".N.", 1)[1].split(".")[0] for x in r["quantized_sample"] if ".N." in x})
    c.kv("quantized", f"{r['linears_quantized']} Linear ({', '.join(g + '.*' for g in groups)})")


def _show_calib(c: Console, e: dict) -> None:
    c.kv("template", TEMPLATES.get(e["template"], e["template"]))
    c.kv("tokens/sample", f"min {e['min']:,} · median {int(e['median']):,} · max {e['max']:,} (truncated: {e['truncated']})")
    if e.get("cached"):
        c.kv("cache", "reused from cache")
    if e["template"] == "raw":
        c.warn("calibration uses raw text: the chat template is missing or drops message content")
    if e["median"] < 64:
        c.warn(f"median calibration sample is only {int(e['median'])} tokens")


def quantize_checkpoint(s: Session, o: QuantizeOptions, ckpt: str, signature: dict) -> dict:
    c, passes = s.console, schemes.CHECKPOINTS[ckpt]["passes"]
    progress, tail = parsers.QuantizeProgress(), deque(maxlen=400)
    st = {"step": 1, "t": s.clock(), "recipe": None, "saved": None, "refused": None, "oneshot_done": False}
    model, mounts = mount_path(o.model, "/model")
    calib, cmounts = mount_path(o.calib, "/calib/" + Path(o.calib).name) if o.calib.endswith(".jsonl") else (o.calib, [])
    key = plan.calib_key(o.model, o.calib, o.samples, o.seq_len)
    s.header(f"[1/4] Loading model · {ckpt}")
    c.kv("source", o.model)

    def show_progress():
        if st["step"] != 3 or st["oneshot_done"]:
            return
        if schemes.data_free(ckpt):  # no calibration, so llmcompressor prints no (N/33) progress
            c.status(f"rounding weights · elapsed {fmt_duration(s.clock() - st['t'])}")
            return
        if not progress.total:
            return
        elapsed, f = s.clock() - st["t"], progress.fraction(passes)
        eta = f" · est. {fmt_duration(elapsed * (1 - f) / f)} left" if f >= 0.02 else ""
        c.status(f"{progress.status_text()} · elapsed {fmt_duration(elapsed)}{eta}")

    def on_event(e):
        kind = e["event"]
        if kind == "step":
            if e["n"] == 1:  # header for step 1 is printed before the container starts
                return
            c.done(s.clock() - st["t"])
            st["step"], st["t"] = e["n"], s.clock()
            if e["n"] == 2:
                s.header("[2/4] Preparing calibration data")
                c.kv("source", f"{o.calib} · {o.samples} samples · seed 42")
            else:
                if schemes.data_free(ckpt):
                    s.header("[2/4] Preparing calibration data")
                    c.kv("calibration", "not needed: weights are rounded, activation scales are computed at runtime")
                s.header(f"[3/4] Quantizing {ckpt}")
                _show_recipe(c, ckpt, st["recipe"])
        elif kind == "config":
            arch = (e.get("architectures") or ["unknown"])[0]
            tag = "(verified)" if arch in VERIFIED_ARCHS else "[unverified architecture]"
            c.kv("architecture", f"{arch} {tag}" + (" · multimodal" if e.get("multimodal") else ""))
            c.kv("dtype", f"{e.get('dtype')} · compute {o.compute_dtype} · quantized: no")
        elif kind == "load":
            c.kv("loaded", f"{e['cls']} · {e['params'] / 1e9:.2f} B params in {fmt_duration(e['seconds'])}")
        elif kind == "linearize":
            c.kv("MoE", f"{e['fused_expert_modules']} fused expert modules linearized")
        elif kind == "recipe":
            st["recipe"] = e
            c.kv("decoder", f"{e['sequential_target']} x {e['num_layers']}" + (" · MTP" if e["mtp"] else ""))
        elif kind == "calib":
            _show_calib(c, e)
        elif kind == "oneshot":
            st["oneshot_done"] = True
            peak = f" · GPU peak {fmt_gb(e['peak_gpu_bytes'])}" if e.get("peak_gpu_bytes") else ""
            c.kv("oneshot", fmt_duration(e["seconds"]) + peak)
        elif kind == "saved":
            st["saved"] = e
            extra = (f" · cast {e['cast']}" if e.get("cast") else "") + (" · MTP saved" if e.get("mtp") else "")
            c.kv("saved", f"checkpoints/{ckpt} · {fmt_gb(o.source_bytes)} → {fmt_gb(e['bytes'])}{extra}")
        elif kind == "refused":
            st["refused"] = e

    def on_line(line):
        tail.append(line)
        if e := parsers.parse_event(line):
            on_event(e)
        elif progress.feed(line):
            show_progress()

    with atomic_dir(s.rd / "checkpoints" / ckpt) as tmp:
        cmd = ["python3", "/qtool/quantize.py", "--model", model, "--checkpoint", ckpt, "--compute-dtype", o.compute_dtype,
               "--calib", calib, "--samples", str(o.samples), "--seq-len", str(o.seq_len),
               "--calib-cache", f"/runs/calib/{key}", "--ignore", ",".join(o.ignore),
               "--signature", json.dumps(signature), "--out", f"/runs/checkpoints/{tmp.name}"]
        code = s.run_container(container_spec(s, s.compress_image, cmd, [*mounts, *cmounts], memory=True),
                     s.rd / "logs" / f"{ckpt}.quantize.log", on_line, on_tick=show_progress)
        if st["refused"]:
            hint = st["refused"].get("hint")
            raise StepFailed(st["refused"]["reason"] + (f" ({hint})" if hint else ""), st["step"])
        if code != 0:
            raise StepFailed(failure_reason(code, tail, s.memory_bytes), st["step"])
        if not st["saved"]:
            raise StepFailed("quantize finished without a saved event (see log)", 3)
    c.done(s.clock() - st["t"])
    return st["saved"]


# ---- eval (step 4) ------------------------------------------------------------------------------------------
@dataclass
class EvalJob:
    label: str
    model: str
    title: str
    env: dict = field(default_factory=dict)
    quantization: str | None = None
    info: list = field(default_factory=list)  # (key, value) rows under the header
    expect_marlin: bool = False


@dataclass
class EvalResult:
    status: str  # OK | FAILED | INVALID
    metrics: list = field(default_factory=list)
    reason: str = ""


def model_args(model: str, quantization: str | None) -> str:
    return f"pretrained={model},{EVAL_ARGS}" + (f",quantization={quantization}" if quantization else "")


def quantization_text(qc: dict | None) -> str:
    if not qc:
        return "none (BF16)"
    group = next(iter((qc.get("config_groups") or {}).values()), {})
    w, a = group.get("weights") or {}, group.get("input_activations")
    if not w:
        return str(qc.get("quant_method"))
    if w.get("type") == "float" and w.get("num_bits") == 4 and w.get("group_size") == 16:
        return f"{qc.get('quant_method')} · NVFP4 W4A{a['num_bits'] if a else 16} g16"
    text = f"{qc.get('quant_method')} · {str(w.get('type')).upper()}{w.get('num_bits')} W{w.get('num_bits')}A{a['num_bits'] if a else 16}"
    return text + (f" g{w['group_size']}" if w.get("group_size") else "") + (" symmetric" if w.get("symmetric") else "")


def save_task_results(rd: Path, label: str, raw: dict, tasks: list[str], meta_for) -> None:
    for task in tasks:
        if task in raw.get("results", {}):
            data = {"qtool": meta_for(task), "results": {task: raw["results"][task]},
                    "n-shot": {task: raw.get("n-shot", {}).get(task)}, "n-samples": {task: raw.get("n-samples", {}).get(task)}}
            plan.result_path(rd, label, task).write_text(json.dumps(data, indent=1))


def eval_job(s: Session, job: EvalJob, tasks: list[str], limit: int | None, mode: str, meta_for,
             baseline: tuple[str, list] | None = None, header: bool = True) -> EvalResult:
    c, t0 = s.console, s.clock()
    if header:
        s.header(f"[4/4] Evaluating {job.title}")
    for k, v in job.info:
        c.kv(k, v)
    model, mounts = container_model(s, job.model)
    tmp = s.rd / "results" / f".tmp-{job.label}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        cmd = ["lm_eval", "--model", "vllm", "--model_args", model_args(model, job.quantization), "--tasks", ",".join(tasks),
               "--batch_size", "auto", "--output_path", f"/runs/results/{tmp.name}", "--log_samples"]
        cmd += ["--apply_chat_template", "--fewshot_as_multiturn"] if mode == "chat" else []
        cmd += ["--limit", str(limit)] if limit else []
        p, tail, shown = parsers.EvalProgress(), deque(maxlen=400), set()

        def on_line(line):
            tail.append(line)
            before = len(p.kernels)
            if not p.feed(line):
                return
            for k in p.kernels[before:]:
                c.kv("kernel", k)
            if p.kv_tokens and "kv" not in shown:
                shown.add("kv")
                c.kv("model memory", f"{p.load_gib} GiB · KV cache {p.kv_tokens:,} tokens")
            if p.downcast and "dc" not in shown:
                shown.add("dc")
                c.kv("downcast", p.downcast.replace("->", " → ") + " (checkpoint stored in a wider dtype)")

        def on_tick():
            if p.started:
                c.status(f"lm_eval      : {p.prompts_done}/{p.prompts_total} prompts")
            elif "kv" not in shown:
                c.status(f"vLLM         : starting (load + compile + CUDA graphs) … {fmt_duration(s.clock() - t0)}")

        code = s.run_container(container_spec(s, s.eval_image, cmd, mounts, env=job.env), s.rd / "logs" / f"{job.label}.eval.log",
                     on_line, on_tick)
        c.end_status()
        path = parsers.find_results_json(tmp) if code == 0 else None
        if path is None:
            shutil.rmtree(tmp, ignore_errors=True)
            reason = failure_reason(code, tail, None) if code != 0 else "lm_eval finished without results_*.json"
            c.failed(reason)
            c.line(f"  log: {s.rd / 'logs' / f'{job.label}.eval.log'}")
            return EvalResult("FAILED", reason=reason)
        try:
            raw = json.loads(path.read_text())
            metrics = parsers.metrics_from(raw["results"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            shutil.rmtree(tmp, ignore_errors=True)
            reason = f"unreadable lm_eval results: {type(e).__name__}"
            c.failed(reason)
            c.line(f"  log: {s.rd / 'logs' / f'{job.label}.eval.log'}")
            return EvalResult("FAILED", reason=reason)
        c.kv("lm_eval", f"{p.prompts_done}/{p.prompts_total} prompts")
        for line in p.table:
            c.line("  " + line)
        save_task_results(s.rd, job.label, raw, tasks, meta_for)
        raw_dir = s.rd / "results" / "raw" / job.label
        shutil.rmtree(raw_dir, ignore_errors=True)
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(raw_dir)
        status = "OK"
        if job.expect_marlin and not any(k.startswith("MarlinLinearKernel") for k in p.kernels):
            status = "INVALID"
            c.warn("INVALID: no MarlinLinearKernel line in the vLLM log, so W4A8 is not proven (score shown anyway)")
        if baseline:
            name, base_metrics = baseline
            mine = parsers.primary(metrics)
            for task, bm in parsers.primary(base_metrics).items():
                if m := mine.get(task):
                    delta, rec = report.vs_baseline(m, bm)
                    c.kv(f"vs {name}", f"{m.label} {bm.value:.3f} → {m.value:.3f} · {delta} · recovery {rec}")
        c.done(s.clock() - t0)
        return EvalResult(status, metrics)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def baseline_row(s: Session, label: str, model: str, title: str, tasks: list[str], limit: int | None, mode: str,
                 weight_bytes: int | None) -> report.Row:
    c = s.console
    s.header(f"[4/4] Evaluating {title}")
    skip = preflight.check_eval_fit(title, weight_bytes, s.gpu)
    if skip:
        c.warn(skip + " → Δ / recovery n/a")
        return report.Row(label, "SKIPPED", size_bytes=weight_bytes, note="does not fit")
    metas = {t: plan.eval_meta(model, t, limit, s.eval_image, mode, s.gpu.name) for t in tasks}
    metrics, todo = [], []
    for t in tasks:
        path = plan.result_path(s.rd, label, t)
        valid, why = plan.cached_result(path, metas[t])
        c.kv("cache", f"{t}: {why}")
        if valid:
            metrics += parsers.read_results(path)["metrics"]
        else:
            todo.append(t)
    if not todo:
        c.done(0)
        return report.Row(label, "OK", parsers.primary(metrics), weight_bytes)
    res = eval_job(s, EvalJob(label, model, title, info=[("model", model)]), todo, limit, mode, metas.get, header=False)
    return report.Row(label, res.status, parsers.primary(metrics + res.metrics), weight_bytes, res.reason)


def print_summary(s: Session, rows: list[report.Row], tasks: list[str], limit: int | None, baseline: str | None,
                  meta: dict) -> None:
    c = s.console
    for task in tasks:
        c.header(report.summary_title(task, rows, limit), right=f"total {fmt_duration(s.clock() - s.started)}")
        for line in report.summary_lines(rows, task, baseline):
            c.line(line)
    path = s.rd / "results" / "summary.json"
    report.write_summary(path, rows, meta)
    c.line(f"  results: {path}")


# ---- whole runs ---------------------------------------------------------------------------------------------
@dataclass
class QuantizeRun:
    model: str
    schemes: list[str]
    states: list[plan.CheckpointState]
    signatures: dict[str, dict]
    options: QuantizeOptions
    params: int | None
    tasks: list[str]
    limit: int | None
    mode: str
    no_eval: bool
    on_error: str
    meta: dict


def run_quantize(s: Session, r: QuantizeRun) -> int:
    c = s.console
    ok, sizes, failed, abort = {}, {}, False, False
    for st in r.states:
        path = s.rd / "checkpoints" / st.name
        if st.action == "stale":
            raise ValueError(f"checkpoint {st.name} is stale; refuse before calling run_quantize")
        if st.action == "reuse":
            ok[st.name], sizes[st.name] = True, dir_bytes(path)
            continue
        try:
            sizes[st.name] = quantize_checkpoint(s, r.options, st.name, r.signatures[st.name])["bytes"]
            ok[st.name] = True
        except StepFailed as e:
            ok[st.name], failed = False, True
            c.failed(e.reason)
            c.line(f"  log: {s.rd / 'logs' / f'{st.name}.quantize.log'}")
            if e.step is not None and e.step < 3:
                c.line("  stopping all schemes: model loading / calibration is shared by every checkpoint")
                abort = True
                break
            if r.on_error == "stop":
                abort = True
                break
    if r.no_eval or abort:
        c.header("Summary · checkpoints", right=f"total {fmt_duration(s.clock() - s.started)}")
        for st in r.states:
            state = "OK" if ok.get(st.name) else ("FAILED" if st.name in ok else "NOT RUN")
            c.line(f"  {st.name:<12}{state:<9}{fmt_gb(sizes.get(st.name)):<10}{s.rd / 'checkpoints' / st.name}")
        return 1 if failed else 0

    rows = [baseline_row(s, "bf16", r.model, "bf16 baseline", r.tasks, r.limit, r.mode,
                         r.params * 2 if r.params else None)]
    base = ("bf16", [m for m in rows[0].metrics.values()]) if rows[0].status == "OK" else None
    if rows[0].status == "FAILED":
        failed = True
    for name in r.schemes:
        if failed and r.on_error == "stop":
            break
        sc = schemes.SCHEMES[name]
        if not ok.get(sc.checkpoint):
            rows.append(report.Row(name, "SKIPPED", note=f"{sc.checkpoint} failed"))
            continue
        path = s.rd / "checkpoints" / sc.checkpoint
        shared = sum(1 for x in r.schemes if schemes.SCHEMES[x].checkpoint == sc.checkpoint) > 1
        qc = json.loads((path / "config.json").read_text()).get("quantization_config")
        model, weight = str(path), sizes[sc.checkpoint]
        info = [("checkpoint", f"checkpoints/{sc.checkpoint}" + (" (shared)" if shared else "")),
                ("quantization", quantization_text(qc))]
        if sc.env:
            info.append(("runtime env", " ".join(f"{k}={v}" for k, v in sc.env.items())))
        skip = preflight.check_eval_fit(name, weight, s.gpu)
        if skip:
            s.header(f"[4/4] Evaluating {name}")
            c.warn(skip)
            rows.append(report.Row(name, "SKIPPED", size_bytes=weight, note="does not fit"))
            continue
        metas = lambda t, name=name: plan.eval_meta(name, t, r.limit, s.eval_image, r.mode, s.gpu.name)  # noqa: E731
        job = EvalJob(name, model, name, sc.env, None, info, expect_marlin=bool(sc.env))
        res = eval_job(s, job, r.tasks, r.limit, r.mode, metas, baseline=base)
        rows.append(report.Row(name, res.status, parsers.primary(res.metrics), weight, "shared" if shared else ""))
        if res.status != "OK":
            failed = True
            if r.on_error == "stop":
                break
    print_summary(s, rows, r.tasks, r.limit, "bf16", r.meta)
    return 1 if failed else 0


@dataclass
class EvalRun:
    model: str
    label: str
    tasks: list[str]
    limit: int | None
    mode: str
    env: dict
    quantization: str | None
    info: list
    weight_bytes: int | None
    baseline: str | None
    baseline_bytes: int | None
    meta: dict


def run_eval(s: Session, r: EvalRun) -> int:
    rows, base = [], None
    if r.baseline:
        rows.append(baseline_row(s, "baseline", r.baseline, f"baseline {r.baseline}", r.tasks, r.limit, r.mode,
                                 r.baseline_bytes))
        if rows[0].status == "OK":
            base = ("baseline", list(rows[0].metrics.values()))
    skip = preflight.check_eval_fit(r.label, r.weight_bytes, s.gpu)
    if skip:
        s.header(f"[4/4] Evaluating {r.label}")
        s.console.warn(skip)
        rows.append(report.Row(r.label, "SKIPPED", size_bytes=r.weight_bytes, note="does not fit"))
    else:
        metas = lambda t: plan.eval_meta(r.model, t, r.limit, s.eval_image, r.mode, s.gpu.name)  # noqa: E731
        job = EvalJob(r.label, r.model, r.label, r.env, r.quantization, r.info, expect_marlin=bool(r.env))
        res = eval_job(s, job, r.tasks, r.limit, r.mode, metas, baseline=base)
        rows.append(report.Row(r.label, res.status, parsers.primary(res.metrics), r.weight_bytes))
    print_summary(s, rows, r.tasks, r.limit, "baseline" if r.baseline else None, r.meta)
    return 0 if all(row.status in ("OK", "SKIPPED") for row in rows) else 1


def label_for(model: str, act: str | None, quantization: str | None) -> str:
    key = re.sub(r"[^\w.-]", "_", plan.model_key(model))
    return key + (f"+act-{act}" if act else "") + (f"+{quantization}-online" if quantization else "")
