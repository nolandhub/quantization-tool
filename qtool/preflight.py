# ABOUTME: Preflight checks (spec §7A): parse nvidia-smi / meminfo / cgroup text, pick the GPU, and turn model size,
# ABOUTME: SM and options into findings (error = exit 2 before anything runs, warn = print and continue).
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from qtool import schemes
from qtool.console import fmt_gb

GPU_QUERY = ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"]
UUID_QUERY = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
APPS_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
MIB = 1024 ** 2
GIB = 1024 ** 3
# Coefficients × BF16 size (params × 2 bytes), measured in the spike (Q3, Q6, Q9)
QUANTIZE_VRAM = {"bfloat16": 0.6, "float32": 1.0}
QUANTIZE_RAM = {"bfloat16": 1.6, "float32": 3.5}
EVAL_UTIL = 0.85
EVAL_MARGIN = 1 * GIB  # CUDA graphs, activations and a minimum of KV cache on top of the weights


class PreflightError(Exception):
    pass


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    cc: str
    used_mib: int
    total_mib: int
    free_mib: int


@dataclass(frozen=True)
class GpuApp:
    gpu_index: int
    pid: int
    process: str
    used_mib: int


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warn"
    message: str


def _rows(text: str) -> list[list[str]]:
    return [[c.strip() for c in line.split(",")] for line in text.splitlines() if line.strip()]


def parse_gpus(text: str) -> list[Gpu]:
    return [Gpu(int(r[0]), r[1], r[2], int(r[3]), int(r[4]), int(r[5])) for r in _rows(text)]


def parse_apps(apps_text: str, uuid_text: str) -> list[GpuApp]:
    index_of = {r[1]: int(r[0]) for r in _rows(uuid_text)}
    return [GpuApp(index_of.get(r[0], -1), int(r[1]), r[2], int(r[3])) for r in _rows(apps_text)]


def query_gpus(run=subprocess.run) -> tuple[list[Gpu], list[GpuApp]]:
    def out(cmd):
        r = run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise PreflightError(f"nvidia-smi failed: {(r.stderr or r.stdout).strip()}")
        return r.stdout

    return parse_gpus(out(GPU_QUERY)), parse_apps(out(APPS_QUERY), out(UUID_QUERY))


def container_id_from_cgroup(text: str) -> str | None:
    m = re.search(r"docker[-/]([0-9a-f]{64})", text)
    return m[1] if m else None


def process_label(app: GpuApp, names_by_id: dict[str, str], read=None) -> str:
    """'pid 123 (container trt-export-server)' when the PID runs in a container, else 'pid 123 python'."""
    read = read or (lambda pid: Path(f"/proc/{pid}/cgroup").read_text())
    try:
        cid = container_id_from_cgroup(read(app.pid))
    except OSError:
        cid = None
    if cid and cid in names_by_id:
        return f"pid {app.pid} (container {names_by_id[cid]})"
    return f"pid {app.pid} {app.process}"


def gpu_table(gpus: list[Gpu], apps: list[GpuApp], label=lambda a: f"pid {a.pid} {a.process}") -> list[str]:
    out = ["  GPU  name                              SM    free / total     in use by"]
    for g in gpus:
        users = ", ".join(f"{label(a)} {a.used_mib} MiB" for a in apps if a.gpu_index == g.index) or "-"
        out.append(f"  {g.index:<4} {g.name[:32]:<33} {g.cc:<5} {g.free_mib / 1024:5.1f} / {g.total_mib / 1024:4.1f} GiB  {users}")
    return out


def select_gpu(gpus: list[Gpu], requested: int | None, apps: list[GpuApp] = ()) -> Gpu:
    if not gpus:
        raise PreflightError("no NVIDIA GPU found by nvidia-smi")
    if requested is None:
        if len(gpus) == 1:
            return gpus[0]
        raise PreflightError("\n".join([f"this machine has {len(gpus)} GPUs: choose one with --gpu N", *gpu_table(gpus, apps)]))
    for g in gpus:
        if g.index == requested:
            return g
    raise PreflightError(f"--gpu {requested} does not exist (GPUs: {', '.join(str(g.index) for g in gpus)})")


def mem_available_bytes(meminfo_text: str) -> int:
    m = re.search(r"^MemAvailable:\s+(\d+) kB", meminfo_text, re.M)
    if not m:
        raise PreflightError("MemAvailable not found in /proc/meminfo")
    return int(m[1]) * 1024


def quantize_needs(params: int, dtype: str) -> tuple[int, int]:
    size = params * 2
    return int(QUANTIZE_VRAM[dtype] * size), int(QUANTIZE_RAM[dtype] * size)


def eval_budget(total_mib: int) -> int:
    return int(EVAL_UTIL * total_mib * MIB) - EVAL_MARGIN


def check_quantize(gpu: Gpu, params: int | None, dtype: str, ram_available: int, schemes_: list[str],
                   no_eval: bool, thresholds: bool = True) -> list[Finding]:
    """thresholds=False is --skip-preflight: VRAM / RAM limits are the developer's call."""
    out = []
    q = schemes.support("quantize", gpu.cc)
    if q.status == schemes.WARN:
        out.append(Finding("warn", f"quantize on SM {gpu.cc}: {q.note}"))
    if thresholds and params is None:
        out.append(Finding("warn", "model size unknown: VRAM / RAM thresholds not checked"))
    elif thresholds:
        vram, ram = quantize_needs(params, dtype)
        free = gpu.free_mib * MIB
        if free < vram:
            out.append(Finding("error", f"GPU {gpu.index} has {fmt_gb(free)} free VRAM, quantize needs {fmt_gb(vram)} "
                                        f"({QUANTIZE_VRAM[dtype]} x {fmt_gb(params * 2)} {dtype})"))
        if ram_available < ram:
            out.append(Finding("error", f"host has {fmt_gb(ram_available)} RAM available, quantize needs {fmt_gb(ram)} "
                                        f"({QUANTIZE_RAM[dtype]} x {fmt_gb(params * 2)} {dtype})"))
    if not no_eval:
        out += check_eval_sm(gpu, schemes_)
    return out


def check_eval_sm(gpu: Gpu, schemes_: list[str], baseline: bool = True) -> list[Finding]:
    out = []
    if schemes.sm_column(gpu.cc) == "7.0":
        return [Finding("error", f"eval cannot run on SM {gpu.cc} (vLLM 0.28 has no SM 7.0 kernels): add --no-eval, "
                                 "then run `qtool eval` on a GPU with SM >= 7.5")]
    rows = [("baseline", "bf16 baseline")] if baseline else []
    rows += [(schemes.SCHEMES[s].row, s) for s in schemes_]
    for row, name in rows:
        s = schemes.support(row, gpu.cc)
        tag = "" if s.verified else " [per vLLM docs, unverified]"
        if s.status == schemes.NO:
            alt = schemes.alternative(name, gpu.cc) if name in schemes.SCHEMES else None
            out.append(Finding("error", f"{name} is not supported on SM {gpu.cc}: {s.note}." + (f" Use {alt}." if alt else "")))
        elif s.status == schemes.WARN:
            out.append(Finding("warn", f"{name} on SM {gpu.cc} {s.note}{tag}"))
        elif tag:
            out.append(Finding("warn", f"{name} on SM {gpu.cc}{tag}"))
    return out


def check_eval_fit(label: str, weight_bytes: int | None, gpu: Gpu) -> str | None:
    """Reason to skip an eval whose weights cannot fit vLLM's memory budget, else None."""
    if weight_bytes is None:
        return None
    budget = eval_budget(gpu.total_mib)
    if weight_bytes > budget:
        return f"{label} does not fit: {fmt_gb(weight_bytes)} weights > {fmt_gb(budget)} budget " \
               f"({EVAL_UTIL} x {gpu.total_mib / 1024:.1f} GiB - 1 GiB)"
    return None


def check_eval_free(gpu: Gpu) -> list[Finding]:
    need = int(EVAL_UTIL * gpu.total_mib)
    if gpu.free_mib < need:
        return [Finding("error", f"vLLM needs {need / 1024:.1f} GiB free on GPU {gpu.index} "
                                 f"(gpu_memory_utilization={EVAL_UTIL}), only {gpu.free_mib / 1024:.1f} GiB free")]
    return []


def is_int4_sym_group(qc: dict | None) -> tuple[bool, str]:
    """--act needs an INT4 symmetric group-wise compressed-tensors checkpoint; returns (ok, what it actually is)."""
    if not qc:
        return False, "not quantized"
    groups = qc.get("config_groups") or {}
    w = next(iter(groups.values()), {}).get("weights") or {}
    kind = f"{qc.get('quant_method')} {w.get('type')}{w.get('num_bits')} symmetric={w.get('symmetric')} " \
           f"strategy={w.get('strategy')}"
    ok = (qc.get("quant_method") == "compressed-tensors" and w.get("num_bits") == 4 and w.get("type") == "int"
          and w.get("symmetric") is True and w.get("strategy") == "group")
    return ok, kind


def check_act(gpu: Gpu, qc: dict | None) -> list[Finding]:
    out = []
    if schemes.sm_column(gpu.cc) not in ("8.9", "12.x"):
        out.append(Finding("error", f"--act needs SM 8.9 or 12.0 (Marlin W4A8), GPU {gpu.index} is SM {gpu.cc}"))
    ok, kind = is_int4_sym_group(qc)
    if not ok:
        out.append(Finding("error", f"--act needs an INT4 symmetric group-wise checkpoint, this model is {kind}"))
    return out


class RunLock:
    """~/qtool/.lock holding the PID; a stale lock (dead PID) is taken over."""

    def __init__(self, path):
        self.path = Path(path)

    def _holder(self) -> int:
        try:
            return int(self.path.read_text().strip() or 0)
        except (OSError, ValueError):
            return 0

    def acquire(self) -> None:
        pid = self._holder()
        if pid and pid != os.getpid() and _alive(pid):
            raise PreflightError(f"another qtool run (pid {pid}) is active")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))

    def release(self) -> None:
        if self._holder() == os.getpid():
            self.path.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
