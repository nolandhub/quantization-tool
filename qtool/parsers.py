# ABOUTME: Parsers for real tool output: @qtool JSON events, llmcompressor progress (tqdm / loguru), vLLM and
# ABOUTME: lm_eval eval logs, lm_eval results JSON and known error lines. Every regex comes from a spike fixture.
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

QTOOL_MARK = "@qtool "
LINE_SPLIT_RE = re.compile(r"\r\n|\r|\n")


def split_lines(text: str) -> list[str]:
    return [line for line in LINE_SPLIT_RE.split(text) if line]


def parse_event(line: str) -> dict | None:
    # tqdm writes "\r" without a newline, so an @qtool line can arrive glued behind progress-bar text
    i = line.find(QTOOL_MARK)
    if i < 0:
        return None
    try:
        obj = json.loads(line[i + len(QTOOL_MARK):])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "event" in obj else None


# ---- llmcompressor -------------------------------------------------------------------------------------------
SUBGRAPH_RE = re.compile(r"\((\d+)/(\d+)\): (Calibrating|Propagating)(?::\s+\d+%\|[^|]*\|\s*(\d+)/(\d+))?")
MODULE_RE = re.compile(r"\| compress_module_list \| INFO - Quantizing (\S+) using (\d+) samples")
GPTQ_TIME_RE = re.compile(r"\| GPTQ \| METRIC - time ([\d.]+)s")
SMOOTH_RE = re.compile(r"_apply_smoothing \| INFO - Smoothing with (\S+)")
LINEARIZE_RE = re.compile(r"MoE is being linearized")


@dataclass
class QuantizeProgress:
    subgraph: int = 0
    total: int = 0
    phase: str = ""
    done: int = 0
    samples: int = 0
    pass_no: int = 1
    module: str | None = None
    gptq_modules: int = 0
    gptq_seconds: float = 0.0
    smoothed: int = 0
    linearized: bool = False

    def feed(self, line: str) -> bool:
        if m := SUBGRAPH_RE.search(line):
            n = int(m[1])
            if n < self.subgraph:  # SmoothQuant + GPTQ: the second pass restarts at (1/N)
                self.pass_no += 1
            self.subgraph, self.total, self.phase = n, int(m[2]), m[3]
            self.done, self.samples = (int(m[4]), int(m[5])) if m[4] else (0, 0)
            return True
        if m := MODULE_RE.search(line):
            self.module = m[1]
            return True
        if m := GPTQ_TIME_RE.search(line):
            self.gptq_modules += 1
            self.gptq_seconds += float(m[1])
            return True
        if SMOOTH_RE.search(line):
            self.smoothed += 1
            return True
        if LINEARIZE_RE.search(line):
            self.linearized = True
            return True
        return False

    def fraction(self, passes: int) -> float:
        if not self.total:
            return 0.0
        within = self.done / self.samples if self.samples else 0.0
        half = 0.5 * within if self.phase == "Calibrating" else 0.5 + 0.5 * within
        return min(1.0, ((self.pass_no - 1) * self.total + self.subgraph - 1 + half) / (passes * self.total))

    def status_text(self) -> str:
        if not self.total:
            return "starting"
        p = f"pass {self.pass_no} · " if self.pass_no > 1 else ""
        s = f" {self.done}/{self.samples}" if self.samples else ""
        return f"{p}({self.subgraph}/{self.total}) {self.phase}{s}"


# ---- vLLM / lm_eval -----------------------------------------------------------------------------------------
KERNEL_RE = re.compile(r"\b(?:Using|Selected) (\w+Kernel) for (\w+)")
LOAD_RE = re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")
KV_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
DOWNCAST_RE = re.compile(r"Downcasting torch\.(\w+) to torch\.(\w+)")
PROMPTS_RE = re.compile(r"Processed prompts:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")


@dataclass
class EvalProgress:
    kernels: list[str] = field(default_factory=list)
    load_gib: float | None = None
    load_seconds: float | None = None
    kv_tokens: int | None = None
    downcast: str | None = None
    prompts_done: int = 0
    prompts_total: int = 0
    table: list[str] = field(default_factory=list)
    _in_table: bool = False

    def feed(self, line: str) -> bool:
        if self._in_table:
            if line.startswith("|"):
                self.table.append(line.rstrip())
                return True
            self._in_table = False
        if line.startswith("|Tasks|"):
            self.table, self._in_table = [line.rstrip()], True
            return True
        if m := KERNEL_RE.search(line):
            kernel = f"{m[1]} for {m[2]}"
            if kernel not in self.kernels:
                self.kernels.append(kernel)
            return True
        if m := LOAD_RE.search(line):
            self.load_gib, self.load_seconds = float(m[1]), float(m[2])
            return True
        if m := KV_RE.search(line):
            self.kv_tokens = int(m[1].replace(",", ""))
            return True
        if m := DOWNCAST_RE.search(line):
            self.downcast = f"{m[1]}->{m[2]}"
            return True
        if m := PROMPTS_RE.search(line):
            self.prompts_done, self.prompts_total = int(m[1]), int(m[2])
            return True
        return False

    @property
    def started(self) -> bool:
        return self.prompts_total > 0


# ---- errors -------------------------------------------------------------------------------------------------
KNOWN_ERRORS = [
    (re.compile(r"exceeds available Mamba cache blocks"), "hybrid model: max_num_seqs is larger than the Mamba cache"),
    (re.compile(r"No available memory for the cache blocks"),
     "weights leave no room for the KV cache at gpu_memory_utilization=0.85"),
    (re.compile(r"OutOfMemoryError|CUDA out of memory"), "GPU out of memory: free the GPU or lower --calib-seq-len"),
    (re.compile(r"\b(?:ImportError|ModuleNotFoundError): "), "a Python package is missing in the container image"),
    (re.compile(r"GatedRepoError|401 Client Error|403 Client Error"), "gated or private model: export HF_TOKEN"),
    (re.compile(r"No space left on device"), "disk full"),
]
EXCEPTION_LINE_RE = re.compile(r"^\s*(?:[\w.]+\.)?\w*(?:Error|Exception)\b")


def find_error(lines: list[str]) -> tuple[str, str] | None:
    """(original line, hint) for the last known error; else the last exception line with an empty hint."""
    for line in reversed(lines):
        for rx, hint in KNOWN_ERRORS:
            if rx.search(line):
                return line.strip(), hint
    for line in reversed(lines):
        if EXCEPTION_LINE_RE.match(line):
            return line.strip(), ""
    return None


# ---- lm_eval results ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Metric:
    task: str
    name: str
    filter: str
    value: float
    stderr: float | None

    @property
    def label(self) -> str:
        return f"{self.task} {self.filter}" if self.filter != "none" else self.task


def metrics_from(results: dict) -> list[Metric]:
    out = []
    for task, vals in results.items():
        for key, value in vals.items():
            if "," not in key or "_stderr," in key or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            name, flt = key.split(",", 1)
            se = vals.get(f"{name}_stderr,{flt}")
            out.append(Metric(task, name, flt, float(value), float(se) if isinstance(se, (int, float)) else None))
    return out


def read_results(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    return {"metrics": metrics_from(d["results"]),
            "n_samples": {t: v.get("effective") for t, v in d.get("n-samples", {}).items()},
            "n_shot": d.get("n-shot", {})}


def primary(metrics: list[Metric]) -> dict[str, Metric]:
    """First metric lm_eval lists per task (gsm8k: exact_match strict-match)."""
    out: dict[str, Metric] = {}
    for m in metrics:
        out.setdefault(m.task, m)
    return out


def find_results_json(root: Path) -> Path | None:
    found = sorted(Path(root).rglob("results_*.json"), key=lambda p: p.name)
    return found[-1] if found else None
