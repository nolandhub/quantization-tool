# qtool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây `qtool` — CLI 2 lệnh con (`quantize`, `eval`) gói 4 bước LLM Compressor của vLLM, log console từng bước bằng tiếng Anh, mọi số liệu parse từ output thật.

**Architecture:** CLI Python trên host (chỉ thư viện chuẩn) điều khiển 2 Docker image: `qtool-compress` (step 1–3, script `images/scripts/quantize.py`) và `qtool-eval` (step 4, `lm_eval --model vllm`). Script trong container in sự kiện `@qtool {json}`; host parse cùng log thư viện (tqdm, loguru, vLLM, lm_eval) theo regex lấy từ fixture spike.

**Tech Stack:** Python ≥ 3.10 stdlib (host), pytest (dev, chạy bằng `uv`), Docker + NVIDIA Container Toolkit, torch 2.13 / llmcompressor 0.13.0 / transformers 5.14.1 (compress), vLLM 0.28 / lm-eval 0.4.13 (eval).

**Spec:** `docs/specs/2026-10-30-qtool-design.md` (bản 3.1) · kết quả spike: `docs/spike/2026-10-30-findings.md`

**Trạng thái code trong plan:** mọi khối code đã chạy thật ngày 2026-11-01 — test host 130 passed (`filterwarnings = error`), test container 17 passed trong image compress (CPU), smoke thật trên rtx4070: `qtool quantize` model tí hon (W4A16 + W8A8, `--no-eval`) và `qtool eval --act fp8 --baseline` với vLLM thật. Plan được sinh từ chính các file đó, nên chép nguyên văn.

## Global Constraints

- Host: chỉ thư viện chuẩn Python ≥ 3.10; không `pip install` gì cho runtime. Dev: `uv run --no-project --with pytest python -m pytest -q` (máy local và rtx4070 đều không có `python3-venv`, dùng `uv`).
- Mọi file code bắt đầu bằng 2 dòng `# ABOUTME:`. Commit **không** có dòng attribution AI (không `Co-Authored-By`, không "Generated with").
- Không bao giờ `--no-verify`. TDD: test trước, thấy fail, code, thấy pass, commit.
- Output test phải sạch: pytest chạy với `filterwarnings = ["error", ...]` (cảnh báo = lỗi).
- Tool **không** dừng / bật container hay service nào; preflight chỉ kiểm và báo.
- Container chạy **root**; sau mỗi container tool `chown -R <uid>:<gid> /runs /qtool-home` (Nhân chốt 01/11: HF cache rtx4070 thuộc root).
- Phiên bản ghim: torch 2.13.0, llmcompressor 0.13.0, transformers 5.14.1 (compress, `cu129` / `cu126`); `vllm/vllm-openai:v0.28.0-cu129-ubuntu2404` + `lm-eval[api]==0.4.13` + `huggingface-hub<2.0` + `num2words` (eval).
- Eval cố định: `max_model_len=4096,gpu_memory_utilization=0.85,max_num_seqs=64,enable_thinking=False`, chế độ chat (`--apply_chat_template --fewshot_as_multiturn`) trừ khi chat template rơi nội dung → completion.
- Ngưỡng preflight (× số tham số × 2 byte): quantize BF16 VRAM 0.6 / RAM 1.6, FP32 VRAM 1.0 / RAM 3.5; eval: weight ≤ 0.85 × VRAM − 1 GiB.
- Exit code: 0 ok · 1 có scheme FAILED / INVALID · 2 lỗi tham số / preflight · 130 Ctrl-C.
- Máy chạy GPU / Docker: `rtx4070` (RTX 4070 Ti SUPER 16 GB, SM 8.9). Đồng bộ: `./sync.sh push`. GPU đang có `trt-export-server` (~210 MiB) — không đụng.

## Khác với spec (đã kiểm chứng khi viết plan, ghi lại để reviewer biết)

- Spec 3.1 đã cập nhật: ngưỡng RAM BF16 1.6× (Phi-4-mini đo 1.6×), size = số tham số × 2 byte, script container ở `images/scripts/` (mount read-only, sửa script không phải build lại image), thêm `qtool/images.py`, `qtool/flow.py`, `bin/qtool`, container probe ở preflight, container chạy root + chown.
- `INVALID` (`--act` mà log vLLM không có `MarlinLinearKernel`) tính như thất bại → exit 1.
- Thứ tự in: preflight → kế hoạch → chạy (khớp minh hoạ §8). `--dry-run` in kế hoạch mà không probe (cache baseline không kiểm).
- Lỗi ở step 1–2 của checkpoint đầu (load / calib dùng chung) dừng toàn bộ bất kể `--on-error` (spec §7B).

## File Structure

```
qtool/                      host package (stdlib only)
  __init__.py  __main__.py  version, `python -m qtool`
  cli.py                    argparse, System (side effects), cmd_quantize / cmd_eval, exit codes
  schemes.py                5 schemes, 2 checkpoints, SM capability matrix, compute dtype
  console.py                header / kv / done / status line, durations, sizes
  parsers.py                @qtool events, llmcompressor / vLLM / lm_eval regexes, errors, results JSON
  preflight.py              nvidia-smi / meminfo / cgroup parsing, GPU choice, thresholds, --act checks, lock
  report.py                 Δ pp, recovery %, summary table, summary.json
  plan.py                   runs dir, checkpoint signatures + reuse, baseline cache, plan text
  images.py                 image tags (Dockerfile hash), cu126 / cu129, pins, build argv
  runner.py                 docker argv, streaming lines (\r), raw log, Ctrl-C stop + drain, atomic dirs
  flow.py                   Session, probe, quantize / eval jobs, baseline cache, run_quantize / run_eval
images/compress/Dockerfile  + pytest layer
images/eval/Dockerfile      + num2words layer
images/scripts/             mounted at /qtool in both containers
  recipe.py                 generic recipe derivation (spike Q9)
  quantize.py               steps 1–3 for one checkpoint, @qtool events, calib cache, FP32 cast-back, qtool.json
  probe.py                  preflight metadata: config, params from safetensors headers, chat mode, lm_eval tasks
bin/qtool                   shim (symlink into ~/.local/bin)
scripts/test-container.sh   container-tier tests inside the compress image
sync.sh  README.md  pyproject.toml  .gitignore
tests/                      host tests (fixtures from the spike) + tests/container (tiny random models, CPU)
```

## Chuẩn bị (một lần, trước Task 1)

- [ ] Commit plan + spec 3.1 + `tests/fixtures/tiny-configs/` trên nhánh `spike`, rồi tạo nhánh làm việc:

```bash
git checkout -b v0.1
```

`tests/fixtures/tiny-configs/*.json` là config thật của 4 model đã kiểm chứng (Qwen3.5-4B, Phi-4-mini, OLMoE, SmolVLM2) thu nhỏ còn hidden 128, 2–4 layer (đủ cho group size 128 của W4A16), sinh bằng script tạm ngày 01/11 — test container dựng model ngẫu nhiên từ đó.

---

### Task 1: Khung repo, shim, cấu hình pytest

Dựng package rỗng chạy được `qtool --version` (qua `python -m qtool` và `bin/qtool`), cấu hình pytest (marker, cảnh báo = lỗi), `.gitignore`, `sync.sh`.

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `qtool/__init__.py`, `qtool/__main__.py`, `qtool/cli.py` (bản tối thiểu, Task 14 thay toàn bộ), `bin/qtool`, `sync.sh`, `tests/conftest.py`
- Test: `tests/test_cli.py` (bản tối thiểu, Task 14 thay toàn bộ)

**Interfaces:**
- Produces: `qtool.__version__ == "0.1.0"`; `qtool.cli.main(argv=None) -> int`; fixture pytest `fixtures` → `Path` tới `tests/fixtures`.

- [ ] **Step 1: Write config files** — `.gitignore`

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 2: Write pyproject (pytest config only, no build backend)** — `pyproject.toml`

```toml
[project]
name = "qtool"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not gpu and not container' -p no:cacheprovider"
markers = [
    "container: runs inside the qtool-compress image (scripts/test-container.sh)",
    "gpu: needs a GPU, Docker images and HF models; run by hand on rtx4070",
]
filterwarnings = [
    "error",
    "ignore:`torch.jit.script_method` is deprecated:DeprecationWarning",
]
```

- [ ] **Step 3: Write the shared test fixture** — `tests/conftest.py`

```python
# ABOUTME: Shared pytest fixtures: path to the real spike fixtures under tests/fixtures.
# ABOUTME: Logs and CSVs captured during the spike are the ground truth for parser and preflight tests.
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
```

- [ ] **Step 4: Write the failing test** — `tests/test_cli.py`

```python
# ABOUTME: Tests for qtool.cli: argument validation (exit 2), dry-run, preflight blocking before any container,
# ABOUTME: a full fake quantize run, stale checkpoint, Ctrl-C (exit 130) and `qtool eval --act` checks.
import subprocess
import sys
from pathlib import Path

import pytest

from qtool import __version__, cli

ROOT = Path(__file__).resolve().parents[1]


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"qtool {__version__}"


def test_bin_shim():
    r = subprocess.run([sys.executable, str(ROOT / "bin" / "qtool"), "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == f"qtool {__version__}"
```

- [ ] **Step 5: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_cli.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool'`

- [ ] **Step 6: Write the package** — `qtool/__init__.py`

```python
# ABOUTME: qtool package: CLI wrapping the vLLM LLM Compressor 4-step flow (load, calibrate, quantize, evaluate).
# ABOUTME: Host side uses only the Python standard library; heavy work runs in two Docker images.
__version__ = "0.1.0"
```

- [ ] **Step 7: Write `python -m qtool`** — `qtool/__main__.py`

```python
# ABOUTME: Entry point for `python -m qtool`.
# ABOUTME: Delegates to qtool.cli.main and exits with its return code.
import sys

from qtool.cli import main

sys.exit(main())
```

- [ ] **Step 8: Write the minimal CLI** — `qtool/cli.py`

```python
# ABOUTME: Command-line interface: parsing and validation for `qtool quantize` / `qtool eval`, preflight wiring
# ABOUTME: (GPU, images, probe, thresholds) and exit codes: 0 ok, 1 a scheme failed, 2 usage/preflight error, 130 Ctrl-C.
import argparse

from qtool import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qtool", description="Quantize a model with LLM Compressor and evaluate it in vLLM.")
    p.add_argument("--version", action="version", version=f"qtool {__version__}")
    return p


def main(argv=None) -> int:
    build_parser().parse_args(argv)
    return 0
```

- [ ] **Step 9: Write the shim (then `chmod +x bin/qtool`)** — `bin/qtool`

```python
#!/usr/bin/env python3
# ABOUTME: Shim that runs qtool straight from the repo checkout (symlink it into ~/.local/bin).
# ABOUTME: Puts the repo root on sys.path, then calls qtool.cli.main.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from qtool.cli import main  # noqa: E402

sys.exit(main())
```

- [ ] **Step 10: Write sync.sh (then `chmod +x sync.sh`)** — `sync.sh`

```bash
#!/usr/bin/env bash
# ABOUTME: rsync the qtool repo between this machine and rtx4070:~/qtool (GPU box where images and tests run).
# ABOUTME: Usage: sync.sh push|pull|check   (check = dry-run push listing what would change)
set -euo pipefail
cd "$(dirname "$0")"
REMOTE=${QTOOL_REMOTE:-rtx4070:qtool/}
EXCL=(--exclude .venv --exclude __pycache__ --exclude .pytest_cache)
case "${1:-}" in
  push) rsync -a --delete "${EXCL[@]}" ./ "$REMOTE" ;;
  pull) rsync -a "${EXCL[@]}" --exclude .git "$REMOTE" ./ ;;
  check) rsync -a --delete --dry-run --itemize-changes "${EXCL[@]}" ./ "$REMOTE" ;;
  *) echo "usage: $0 push|pull|check" >&2; exit 2 ;;
esac
```

- [ ] **Step 11: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_cli.py`
Expected: `2 passed`

- [ ] **Step 12: Commit**

```bash
git add .gitignore pyproject.toml tests/conftest.py tests/test_cli.py qtool/__init__.py qtool/__main__.py qtool/cli.py bin/qtool sync.sh
git commit -m "Scaffold qtool package, shim and pytest config"
```

---

### Task 2: schemes.py — scheme, checkpoint, ma trận SM

Bảng tĩnh của spec §3: 5 scheme, 2 checkpoint dùng chung, ma trận khả năng eval theo SM (✓/⚠/✗ + đã kiểm chứng hay chưa), gợi ý scheme thay thế, compute dtype `auto`.

**Files:**
- Create: `qtool/schemes.py`
- Test: `tests/test_schemes.py`

**Interfaces:**
- Produces: `Scheme(name, checkpoint, row, env, quantization)`, `SCHEMES: dict[str, Scheme]`, `CHECKPOINTS[name] -> {preset, recipe, passes}`, `OK/WARN/NO`, `Support(status, note, verified)`, `sm_column(cc) -> str`, `support(row, cc) -> Support`, `alternative(scheme, cc) -> str | None`, `compute_dtype(cc, requested) -> str`, `parse_schemes(text) -> list[str]` (ValueError), `checkpoints_for(schemes) -> dict[str, list[str]]`.

- [ ] **Step 1: Write the failing test** — `tests/test_schemes.py`

```python
# ABOUTME: Tests for qtool.schemes: scheme parsing, checkpoint sharing, SM capability matrix, compute dtype.
# ABOUTME: Matrix expectations are copied from spec §3.
import pytest

from qtool import schemes


def test_parse_schemes_dedupes_and_keeps_order():
    assert schemes.parse_schemes("int4-w4a8-fp8, int4-w4a16,int4-w4a8-fp8") == ["int4-w4a8-fp8", "int4-w4a16"]


def test_parse_schemes_rejects_unknown():
    with pytest.raises(ValueError, match="unknown scheme 'int4-w4a8'. Choose from: int4-w4a16, int4-w4a8-fp8"):
        schemes.parse_schemes("int4-w4a8")


def test_parse_schemes_rejects_empty():
    with pytest.raises(ValueError, match="--scheme is empty"):
        schemes.parse_schemes(" , ")


def test_checkpoints_are_shared_and_fp8_has_none():
    got = schemes.checkpoints_for(["int4-w4a8-fp8", "int4-w4a16", "fp8-w8a8", "int8-w8a8"])
    assert got == {"INT4-W4A16": ["int4-w4a8-fp8", "int4-w4a16"], "INT8-W8A8": ["int8-w8a8"]}


@pytest.mark.parametrize("cc,col", [("7.0", "7.0"), ("6.1", "7.0"), ("7.5", "7.5"), ("8.0", "8.x"), ("8.6", "8.x"),
                                    ("8.9", "8.9"), ("9.0", "9.0"), ("10.0", "10.x"), ("12.0", "12.x")])
def test_sm_column(cc, col):
    assert schemes.sm_column(cc) == col


def test_support_matrix_samples():
    assert schemes.support("int4-w4a8", "8.9") == schemes.Support("ok", "", True)
    assert schemes.support("int8-w8a8", "12.0").status == "no"
    fp8 = schemes.support("fp8-w8a8", "8.6")
    assert (fp8.status, fp8.note, fp8.verified) == ("warn", "runs as W8A16 weight-only (Marlin)", False)
    assert schemes.support("quantize", "7.0") == schemes.Support("warn", "float32 compute, about 10x slower than bfloat16", True)
    assert schemes.support("baseline", "7.0").status == "no"


def test_alternative_prefers_same_width():
    assert schemes.alternative("int8-w8a8", "10.0") == "fp8-w8a8"
    assert schemes.alternative("int4-w4a8-fp8", "9.0") == "int4-w4a16"
    assert schemes.alternative("int4-w4a16", "7.0") is None


@pytest.mark.parametrize("cc,req,want", [("8.9", "auto", "bfloat16"), ("7.0", "auto", "float32"),
                                         ("7.5", "auto", "float32"), ("7.0", "bfloat16", "bfloat16")])
def test_compute_dtype(cc, req, want):
    assert schemes.compute_dtype(cc, req) == want
```

- [ ] **Step 2: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_schemes.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool.schemes'`

- [ ] **Step 3: Write the implementation** — `qtool/schemes.py`

```python
# ABOUTME: Static tables from spec §3: the 5 schemes, the 2 checkpoints they share, and the eval capability
# ABOUTME: matrix per GPU compute capability (SM). Pure data plus lookups; every hardware rule lives here.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scheme:
    name: str
    checkpoint: str | None  # None: no checkpoint, vLLM quantizes the BF16 model online
    row: str  # capability-matrix row
    env: dict = field(default_factory=dict)
    quantization: str | None = None


SCHEMES = {s.name: s for s in [
    Scheme("int4-w4a16", "INT4-W4A16", "int4-w4a16"),
    Scheme("int4-w4a8-fp8", "INT4-W4A16", "int4-w4a8", {"VLLM_MARLIN_INPUT_DTYPE": "fp8"}),
    Scheme("int4-w4a8-int8", "INT4-W4A16", "int4-w4a8", {"VLLM_MARLIN_INPUT_DTYPE": "int8"}),
    Scheme("int8-w8a8", "INT8-W8A8", "int8-w8a8"),
    Scheme("fp8-w8a8", None, "fp8-w8a8", quantization="fp8"),
]}

# llmcompressor preset per checkpoint and how many sequential passes oneshot makes (SmoothQuant adds one)
CHECKPOINTS = {
    "INT4-W4A16": {"preset": "W4A16", "recipe": "GPTQModifier targets=Linear scheme=W4A16", "passes": 1},
    "INT8-W8A8": {"preset": "W8A8", "passes": 2,
                  "recipe": "SmoothQuantModifier smoothing_strength=0.8 + GPTQModifier targets=Linear scheme=W8A8"},
}

OK, WARN, NO = "ok", "warn", "no"
NO_SM70 = (NO, "vLLM 0.28 has no SM 7.0 kernels")
W4A8_SM = (NO, "Marlin W4A8 needs SM 8.9 or 12.0")
NO_INT8 = (NO, "no INT8 kernels on SM >= 10.0 (vLLM docs)")
FP32_SLOW = (WARN, "float32 compute, about 10x slower than bfloat16")
MATRIX = {
    "int4-w4a16": {"7.0": NO_SM70, "7.5": (NO, "INT4 needs SM > 8.0 (vLLM docs)"), "8.x": (OK, ""), "8.9": (OK, ""),
                   "9.0": (OK, ""), "10.x": (OK, ""), "12.x": (OK, "")},
    "int4-w4a8": {"7.0": NO_SM70, "7.5": W4A8_SM, "8.x": W4A8_SM, "8.9": (OK, ""),
                  "9.0": (NO, "Marlin W4A8 is slower than W4A16 on SM 9.0"), "10.x": W4A8_SM, "12.x": (OK, "")},
    "int8-w8a8": {"7.0": NO_SM70, "7.5": (OK, ""), "8.x": (OK, ""), "8.9": (OK, ""), "9.0": (OK, ""),
                  "10.x": NO_INT8, "12.x": NO_INT8},
    "fp8-w8a8": {"7.0": NO_SM70, "7.5": (NO, "FP8 needs SM >= 8.0"), "8.x": (WARN, "runs as W8A16 weight-only (Marlin)"),
                 "8.9": (OK, ""), "9.0": (OK, ""), "10.x": (OK, ""), "12.x": (OK, "")},
    "baseline": {"7.0": NO_SM70, "7.5": (WARN, "vLLM forces FP16 (no BF16 on SM 7.5)"), "8.x": (OK, ""), "8.9": (OK, ""),
                 "9.0": (OK, ""), "10.x": (OK, ""), "12.x": (OK, "")},
    "quantize": {"7.0": FP32_SLOW, "7.5": FP32_SLOW, "8.x": (OK, ""), "8.9": (OK, ""), "9.0": (OK, ""),
                 "10.x": (OK, ""), "12.x": (OK, "")},
}
VERIFIED = {(row, "8.9") for row in MATRIX} | {("quantize", "7.0")}
# Preferred replacement when a scheme cannot run on a GPU (first one that is OK there wins)
ALTERNATIVES = {"int8-w8a8": ["fp8-w8a8", "int4-w4a16"], "fp8-w8a8": ["int8-w8a8", "int4-w4a16"],
                "int4-w4a8": ["int4-w4a16", "int8-w8a8"], "int4-w4a16": ["int8-w8a8", "fp8-w8a8"]}


@dataclass(frozen=True)
class Support:
    status: str
    note: str
    verified: bool


def _cc(cc: str) -> tuple[int, int]:
    major, minor = cc.split(".")
    return int(major), int(minor)


def sm_column(cc: str) -> str:
    major, minor = _cc(cc)
    if major < 7 or (major == 7 and minor < 5):
        return "7.0"
    if major == 7:
        return "7.5"
    if major == 8:
        return "8.9" if minor >= 9 else "8.x"
    if major == 9:
        return "9.0"
    if major in (10, 11):
        return "10.x"
    return "12.x"


def support(row: str, cc: str) -> Support:
    col = sm_column(cc)
    status, note = MATRIX[row][col]
    return Support(status, note, (row, col) in VERIFIED)


def alternative(scheme: str, cc: str) -> str | None:
    for name in ALTERNATIVES.get(SCHEMES[scheme].row, []):
        if support(SCHEMES[name].row, cc).status == OK:
            return name
    return None


def compute_dtype(cc: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "bfloat16" if _cc(cc)[0] >= 8 else "float32"  # never float16: BF16-trained models can overflow it


def parse_schemes(text: str) -> list[str]:
    names = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in SCHEMES:
            raise ValueError(f"unknown scheme '{name}'. Choose from: {', '.join(SCHEMES)}")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("--scheme is empty")
    return names


def checkpoints_for(schemes: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in schemes:
        ckpt = SCHEMES[s].checkpoint
        if ckpt:
            out.setdefault(ckpt, []).append(s)
    return out
```

- [ ] **Step 4: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_schemes.py`
Expected: `19 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_schemes.py qtool/schemes.py
git commit -m "Add scheme table and SM capability matrix"
```

---

### Task 3: console.py — header, dòng key/value, status line

Định dạng console thống nhất: header `══════ [n/4] … ═══ hh:mm:ss`, dòng `  key          : value`, `✓ done in`, `✗ FAILED`, `!` cảnh báo, status line ghi đè trên TTY / tối đa 1 dòng mỗi 30 s khi pipe.

**Files:**
- Create: `qtool/console.py`
- Test: `tests/test_console.py`

**Interfaces:**
- Produces: `fmt_duration(sec) -> str`, `fmt_gb(bytes|None) -> str`, `WIDTH = 64`, `Console(stream, tty, clock, wallclock, status_interval)` với `.line()`, `.header(title, right=None)`, `.kv(key, value)`, `.done(sec)`, `.failed(reason)`, `.warn(text)`, `.status(text)`, `.end_status()`.

- [ ] **Step 1: Write the failing test** — `tests/test_console.py`

```python
# ABOUTME: Tests for qtool.console: duration/size formatting, header layout, rows and the live status line
# ABOUTME: on a TTY (rewritten in place) versus a pipe (throttled to one line per interval).
import io

import pytest

from qtool.console import WIDTH, Console, fmt_duration, fmt_gb


@pytest.mark.parametrize("sec,text", [(5, "5s"), (59.6, "1m 00s"), (527, "8m 47s"), (9605, "2h 40m")])
def test_fmt_duration(sec, text):
    assert fmt_duration(sec) == text


def test_fmt_gb():
    assert fmt_gb(5_200_000_000) == "5.2 GB"
    assert fmt_gb(None) == "unknown"


def make(tty, now=None):
    out = io.StringIO()
    clock = (lambda: now[0]) if now else None
    return out, Console(out, tty=tty, clock=clock or (lambda: 0.0), wallclock=lambda: "21:01:08")


def test_header_and_rows():
    out, c = make(False)
    c.header("[3/4] Quantizing INT4-W4A16")
    c.kv("recipe", "GPTQModifier")
    c.done(527)
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("══════ [3/4] Quantizing INT4-W4A16 ═") and lines[0].endswith("═ 21:01:08")
    assert len(lines[0]) == WIDTH
    assert lines[1] == "  recipe       : GPTQModifier"
    assert lines[2] == "  ✓ done in 8m 47s"


def test_header_with_custom_right_text():
    out, c = make(False)
    c.header("Summary · gsm8k strict-match (full)", right="total 41m 10s")
    assert out.getvalue().rstrip("\n").endswith("═ total 41m 10s")


def test_status_on_tty_rewrites_one_line():
    out, c = make(True)
    c.status("(1/33) Calibrating 10/512")
    c.status("(1/33) Calibrating 20/512")
    c.line("next")
    assert out.getvalue() == "\r\x1b[K  (1/33) Calibrating 10/512\r\x1b[K  (1/33) Calibrating 20/512\nnext\n"


def test_status_when_piped_is_throttled():
    now = [0.0]
    out, c = make(False, now)
    c.status("a")
    now[0] = 10.0
    c.status("b")
    now[0] = 31.0
    c.status("c")
    assert out.getvalue() == "  a\n  c\n"
```

- [ ] **Step 2: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_console.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool.console'`

- [ ] **Step 3: Write the implementation** — `qtool/console.py`

```python
# ABOUTME: Console rendering: step headers, key/value rows, done/fail marks, durations, sizes and a live status line.
# ABOUTME: The status line is rewritten in place on a TTY; when piped, at most one status line per interval is printed.
import sys
import time

WIDTH = 64


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def fmt_gb(nbytes: float | None) -> str:
    return "unknown" if nbytes is None else f"{nbytes / 1e9:.1f} GB"


class Console:
    def __init__(self, stream=None, tty: bool | None = None, clock=time.monotonic, wallclock=None,
                 status_interval: float = 30.0):
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty() if tty is None else tty
        self.clock = clock
        self.wallclock = wallclock or (lambda: time.strftime("%H:%M:%S"))
        self.status_interval = status_interval
        self._status_shown = False
        self._last_status_at: float | None = None

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def line(self, text: str = "") -> None:
        self.end_status()
        self._write(text + "\n")

    def header(self, title: str, right: str | None = None) -> None:
        left, right = f"══════ {title} ", f" {self.wallclock() if right is None else right}"
        self.line(left + "═" * max(3, WIDTH - len(left) - len(right)) + right)

    def kv(self, key: str, value) -> None:
        self.line(f"  {key:<13}: {value}")

    def done(self, seconds: float) -> None:
        self.line(f"  ✓ done in {fmt_duration(seconds)}")

    def failed(self, reason: str) -> None:
        self.line(f"  ✗ FAILED: {reason}")

    def warn(self, text: str) -> None:
        self.line(f"  ! {text}")

    def status(self, text: str) -> None:
        if self.tty:
            self._write("\r\x1b[K  " + text)
            self._status_shown = True
            return
        now = self.clock()
        if self._last_status_at is None or now - self._last_status_at >= self.status_interval:
            self._last_status_at = now
            self._write("  " + text + "\n")

    def end_status(self) -> None:
        if self._status_shown:
            self._write("\n")
            self._status_shown = False
        self._last_status_at = None
```

- [ ] **Step 4: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_console.py`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_console.py qtool/console.py
git commit -m "Add console rendering helpers"
```

---

### Task 4: parsers.py — parse log thật

Mọi regex lấy từ fixture spike (`tests/fixtures/llmcompressor`, `lm_eval`). Dòng `@qtool` tìm ở bất kỳ đâu trong dòng (tqdm viết `\r` không xuống dòng). W8A8 có 2 lượt `(N/33)` (SmoothQuant + GPTQ). Lỗi đã biết (Mamba, KV, OOM, ImportError, 401/403, ENOSPC) ưu tiên hơn dòng exception chung.

**Files:**
- Create: `qtool/parsers.py`
- Test: `tests/test_parsers.py`

**Interfaces:**
- Produces: `split_lines(text)`, `parse_event(line) -> dict | None`, `QuantizeProgress` (`feed(line) -> bool`, `fraction(passes)`, `status_text()`, fields `subgraph,total,phase,done,samples,pass_no,module,gptq_modules,gptq_seconds,smoothed,linearized`), `EvalProgress` (`feed`, `started`, fields `kernels,load_gib,load_seconds,kv_tokens,downcast,prompts_done,prompts_total,table`), `find_error(lines) -> (quote, hint) | None`, `Metric(task,name,filter,value,stderr)` + `.label`, `metrics_from(results)`, `read_results(path)`, `primary(metrics) -> dict[task, Metric]`, `find_results_json(root)`.

- [ ] **Step 1: Write the failing test** — `tests/test_parsers.py`

```python
# ABOUTME: Tests for qtool.parsers against real spike fixtures: llmcompressor progress, vLLM/lm_eval eval logs,
# ABOUTME: known error lines and lm_eval results JSON. Expected numbers were read from the fixtures themselves.
import json

import pytest

from qtool import parsers


def lines_of(fixtures, rel):
    return parsers.split_lines((fixtures / rel).read_text(errors="replace"))


def feed_all(progress, lines):
    for line in lines:
        progress.feed(line)
    return progress


def test_parse_event_finds_mark_anywhere_in_line():
    assert parsers.parse_event('Dispatching: 0%|  |@qtool {"event": "saved", "bytes": 5}') == {"event": "saved", "bytes": 5}
    assert parsers.parse_event("@qtool {not json") is None
    assert parsers.parse_event('@qtool {"no": "event"}') is None
    assert parsers.parse_event("plain line") is None


def test_split_lines_handles_carriage_returns():
    assert parsers.split_lines("a\rb\r\nc\n\nd") == ["a", "b", "c", "d"]


@pytest.mark.parametrize("rel,total,passes,gptq,gsec,smoothed,linearized,module", [
    ("llmcompressor/w4a16-gptq.log", 33, 1, 128, 218.72, 0, False, "model.language_model.layers.31.mlp.down_proj"),
    ("llmcompressor/w8a8-smoothquant-gptq.log", 33, 2, 128, 204.91, 38, False, "model.language_model.layers.31.mlp.down_proj"),
    ("llmcompressor/olmoe-w4a16-generic.log", 17, 1, 3136, 2070.83, 0, True, "model.layers.15.mlp.experts.63.down_proj"),
    ("llmcompressor/smolvlm2-w4a16-generic.log", 25, 1, 168, 195.07, 0, False, "model.text_model.layers.23.mlp.down_proj"),
    ("llmcompressor/w4a16-gptq-v100-fp32.log", 33, 1, 128, 1011.44, 0, False, "model.language_model.layers.31.mlp.down_proj"),
])
def test_quantize_progress_on_real_logs(fixtures, rel, total, passes, gptq, gsec, smoothed, linearized, module):
    p = feed_all(parsers.QuantizeProgress(), lines_of(fixtures, rel))
    assert (p.subgraph, p.total, p.pass_no) == (total, total, passes)
    assert (p.done, p.samples) == (512, 512)
    assert p.gptq_modules == gptq and p.gptq_seconds == pytest.approx(gsec, abs=0.01)
    assert (p.smoothed, p.linearized, p.module) == (smoothed, linearized, module)
    assert p.fraction(passes) == 1.0


def test_fraction_is_monotonic_and_counts_both_passes(fixtures):
    p, seen, at_pass2 = parsers.QuantizeProgress(), [], None
    for line in lines_of(fixtures, "llmcompressor/w8a8-smoothquant-gptq.log"):
        if p.feed(line) and p.total:
            seen.append(p.fraction(2))
            if p.pass_no == 2 and at_pass2 is None:
                at_pass2 = seen[-1]
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert at_pass2 == pytest.approx(0.5, abs=0.01)


def test_status_text():
    p = parsers.QuantizeProgress()
    assert p.status_text() == "starting"
    p.feed("(17/33): Calibrating:  50%|█████     | 256/512 [00:06<00:06, 40.00it/s]")
    assert p.status_text() == "(17/33) Calibrating 256/512"


def test_progress_of_killed_run_stops_early(fixtures):
    p = feed_all(parsers.QuantizeProgress(), lines_of(fixtures, "llmcompressor/w4a16-fp32-oom-killed.log"))
    assert (p.subgraph, p.total, p.gptq_modules) == (2, 33, 0)


@pytest.mark.parametrize("rel,kernels,gib,kv,downcast", [
    ("lm_eval/eval-w4a16-fp8.log", ["MarlinLinearKernel for CompressedTensorsWNA16"], 5.11, 104448, None),
    ("lm_eval/eval-w8a8.log", ["CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8"], 6.29, 82212, None),
    ("lm_eval/eval-fp8-online.log", ["CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod"], 5.09, 124928, None),
    ("lm_eval/eval-bf16-chat.log", [], 8.61, 42422, None),
    ("lm_eval/eval-w4a16-v100.log", ["MarlinLinearKernel for CompressedTensorsWNA16"], 5.11, 103862, "float32->bfloat16"),
    ("lm_eval/eval-phi4mini-w4a16.log", ["MarlinLinearKernel for CompressedTensorsWNA16"], 2.73, 73855, None),
])
def test_eval_progress_on_real_logs(fixtures, rel, kernels, gib, kv, downcast):
    p = feed_all(parsers.EvalProgress(), lines_of(fixtures, rel))
    assert (p.kernels, p.load_gib, p.kv_tokens, p.downcast) == (kernels, gib, kv, downcast)
    assert (p.prompts_done, p.prompts_total) == (50, 50)
    assert p.table[0].startswith("|Tasks|") and len(p.table) == 4
    assert p.table[3].startswith("|     |       |strict-match")


@pytest.mark.parametrize("rel,quote,hint", [
    ("lm_eval/eval-bf16-docs.err-mamba.log", "max_num_seqs (256) exceeds available Mamba cache blocks (124)", "Mamba"),
    ("lm_eval/eval-olmoe-bf16.err-no-kv-memory.log", "No available memory for the cache blocks", "KV cache"),
    ("lm_eval/eval-smolvlm2-bf16.err-num2words.log", "ImportError: Package `num2words` is required", "missing"),
    ("llmcompressor/w4a16-error-no-cc.log", "RuntimeError: Failed to find C compiler", ""),
    ("llmcompressor/w4a16-error-pydantic.log", "ValidationError: 1 validation error for GPTQModifier", ""),
])
def test_find_error(fixtures, rel, quote, hint):
    got = parsers.find_error(lines_of(fixtures, rel))
    assert got is not None and quote in got[0] and hint in got[1]


def test_find_error_none_when_log_just_stops(fixtures):
    assert parsers.find_error(lines_of(fixtures, "llmcompressor/w4a16-fp32-oom-killed.log")) is None


def test_read_results(fixtures):
    path = parsers.find_results_json(fixtures / "lm_eval/out/w4a16-fp8")
    r = parsers.read_results(path)
    first = parsers.primary(r["metrics"])["gsm8k"]
    assert (first.name, first.filter, first.value) == ("exact_match", "strict-match", 0.78)
    assert first.stderr == pytest.approx(0.0592, abs=1e-4) and first.label == "gsm8k strict-match"
    assert [m.filter for m in r["metrics"]] == ["strict-match", "flexible-extract"]
    assert r["n_samples"] == {"gsm8k": 50} and r["n_shot"] == {"gsm8k": 5}


def test_metrics_skip_non_numeric_stderr():
    ms = parsers.metrics_from({"t": {"acc,none": 0.5, "acc_stderr,none": "N/A", "alias": "t"}})
    assert ms == [parsers.Metric("t", "acc", "none", 0.5, None)] and ms[0].label == "t"


def test_find_results_json_missing(tmp_path):
    assert parsers.find_results_json(tmp_path) is None
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "results_2026-01-01.json").write_text(json.dumps({"results": {}}))
    assert parsers.find_results_json(tmp_path).name == "results_2026-01-01.json"
```

- [ ] **Step 2: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_parsers.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool.parsers'`

- [ ] **Step 3: Write the implementation** — `qtool/parsers.py`

```python
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
```

- [ ] **Step 4: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_parsers.py`
Expected: `25 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_parsers.py qtool/parsers.py
git commit -m "Add parsers for llmcompressor, vLLM and lm_eval output"
```

---

### Task 5: preflight.py — GPU, ngưỡng, kiểm tra trước khi chạy

Parse nvidia-smi (CSV thật rtx4070 / Apollo 8× V100), chọn GPU (bắt buộc `--gpu` khi ≥ 2), map PID → tên container qua `/proc/<pid>/cgroup`, ngưỡng VRAM / RAM quantize, ngân sách VRAM eval, kiểm SM, `--act`, lock file.

**Files:**
- Create: `qtool/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `schemes.support/alternative/sm_column/SCHEMES`, `console.fmt_gb`.
- Produces: `PreflightError`, `Gpu(index,name,cc,used_mib,total_mib,free_mib)`, `GpuApp(gpu_index,pid,process,used_mib)`, `Finding(level,message)`, `parse_gpus`, `parse_apps(apps_text, uuid_text)`, `query_gpus(run)`, `container_id_from_cgroup`, `process_label(app, names_by_id, read)`, `gpu_table`, `select_gpu(gpus, requested, apps)`, `mem_available_bytes`, `quantize_needs(params, dtype) -> (vram, ram)`, `eval_budget(total_mib)`, `check_quantize(gpu, params, dtype, ram, schemes, no_eval, thresholds=True)`, `check_eval_sm(gpu, schemes, baseline=True)`, `check_eval_fit(label, weight_bytes, gpu) -> str | None`, `check_eval_free(gpu)`, `is_int4_sym_group(qc)`, `check_act(gpu, qc)`, `RunLock(path).acquire()/.release()`, hằng `GIB`, `MIB`, `EVAL_UTIL`.

- [ ] **Step 1: Write the failing test** — `tests/test_preflight.py`

```python
# ABOUTME: Tests for qtool.preflight on real nvidia-smi fixtures (rtx4070, 8x V100 Apollo): GPU selection,
# ABOUTME: process labels, VRAM/RAM thresholds, eval capability by SM, eval memory budget, --act and the run lock.
import os

import pytest

from qtool import preflight as pf

GB = 10 ** 9


def gpus_of(fixtures, name):
    return pf.parse_gpus((fixtures / f"nvidia-smi/{name}.csv").read_text())


def apps_of(fixtures, name, uuid):
    return pf.parse_apps((fixtures / f"nvidia-smi/{name}.csv").read_text(),
                         (fixtures / f"nvidia-smi/{uuid}.csv").read_text())


def test_parse_rtx4070(fixtures):
    assert gpus_of(fixtures, "rtx4070") == [pf.Gpu(0, "NVIDIA GeForce RTX 4070 Ti SUPER", "8.9", 16, 16376, 15928)]
    assert apps_of(fixtures, "rtx4070-apps", "rtx4070-uuid") == []
    assert apps_of(fixtures, "rtx4070-busy-apps", "rtx4070-uuid") == [pf.GpuApp(0, 3551455, "python", 1524)]


def test_parse_apollo_maps_uuid_to_index(fixtures):
    apps = apps_of(fixtures, "apollo-8gpu-apps", "apollo-8gpu-uuid")
    assert len(gpus_of(fixtures, "apollo-8gpu")) == 8 and len(apps) == 9
    assert pf.GpuApp(4, 2273897, "VLLM::EngineCore", 30706) in apps


def test_single_gpu_is_chosen_without_flag(fixtures):
    assert pf.select_gpu(gpus_of(fixtures, "rtx4070"), None).index == 0


def test_multi_gpu_requires_flag_and_prints_table(fixtures):
    gpus, apps = gpus_of(fixtures, "apollo-8gpu"), apps_of(fixtures, "apollo-8gpu-apps", "apollo-8gpu-uuid")
    with pytest.raises(pf.PreflightError) as e:
        pf.select_gpu(gpus, None, apps)
    msg = str(e.value)
    assert msg.startswith("this machine has 8 GPUs: choose one with --gpu N")
    assert "pid 2273897 VLLM::EngineCore 30706 MiB" in msg
    assert pf.select_gpu(gpus, 3).index == 3
    with pytest.raises(pf.PreflightError, match="--gpu 9 does not exist"):
        pf.select_gpu(gpus, 9)


def test_query_gpus_reports_nvidia_smi_failure():
    class R:
        returncode, stdout, stderr = 9, "", "NVIDIA-SMI has failed"
    with pytest.raises(pf.PreflightError, match="nvidia-smi failed: NVIDIA-SMI has failed"):
        pf.query_gpus(run=lambda *a, **k: R())


def test_container_label_from_cgroup():
    cid = "42feb688a9f21aab9a07719310988c7d71d4e1bded2f9e5d709712f6d86274ac"
    text = f"0::/system.slice/docker-{cid}.scope\n"
    assert pf.container_id_from_cgroup(text) == cid
    assert pf.container_id_from_cgroup("12:pids:/docker/" + cid) == cid
    app = pf.GpuApp(0, 3551455, "python", 1524)
    assert pf.process_label(app, {cid: "trt-export-server"}, read=lambda pid: text) == \
        "pid 3551455 (container trt-export-server)"
    assert pf.process_label(app, {}, read=lambda pid: "0::/user.slice") == "pid 3551455 python"


def test_mem_available():
    assert pf.mem_available_bytes("MemTotal: 30000000 kB\nMemAvailable:   27100000 kB\n") == 27100000 * 1024
    with pytest.raises(pf.PreflightError):
        pf.mem_available_bytes("MemTotal: 1 kB\n")


def test_quantize_thresholds_bf16_pass_on_rtx4070(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    params_4b = 4_539_265_536
    assert pf.check_quantize(gpu, params_4b, "bfloat16", 27 * GB, ["int4-w4a16"], no_eval=False) == []


def test_quantize_fp32_4b_blocked_by_ram(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    found = pf.check_quantize(gpu, 4_539_265_536, "float32", 27 * GB, ["int4-w4a16"], no_eval=False)
    assert [f.level for f in found] == ["error"]
    assert found[0].message == "host has 27.0 GB RAM available, quantize needs 31.8 GB (3.5 x 9.1 GB float32)"


def test_quantize_vram_error_when_gpu_busy(fixtures):
    gpu = gpus_of(fixtures, "apollo-8gpu")[4]  # 1.7 GiB free
    found = pf.check_quantize(gpu, 4_539_265_536, "float32", 700 * GB, ["int4-w4a16"], no_eval=True)
    assert [f.level for f in found] == ["warn", "error"]
    assert found[0].message == "quantize on SM 7.0: float32 compute, about 10x slower than bfloat16"
    assert found[1].message.startswith("GPU 4 has 1.9 GB free VRAM, quantize needs 9.1 GB")


def test_skip_preflight_drops_thresholds_only(fixtures):
    gpu = gpus_of(fixtures, "apollo-8gpu")[4]
    found = pf.check_quantize(gpu, 4_539_265_536, "float32", 1 * GB, ["int4-w4a16"], no_eval=False, thresholds=False)
    assert [f.level for f in found] == ["warn", "error"] and "add --no-eval" in found[1].message


def test_v100_without_no_eval_is_an_error(fixtures):
    gpu = gpus_of(fixtures, "apollo-8gpu")[1]
    found = pf.check_quantize(gpu, 4_539_265_536, "float32", 700 * GB, ["int4-w4a16"], no_eval=False)
    assert found[-1].level == "error" and "add --no-eval" in found[-1].message


def test_eval_sm_errors_and_warnings():
    sm10 = pf.Gpu(0, "B200", "10.0", 0, 180000, 180000)
    msgs = [(f.level, f.message) for f in pf.check_eval_sm(sm10, ["int8-w8a8"])]
    assert ("error", "int8-w8a8 is not supported on SM 10.0: no INT8 kernels on SM >= 10.0 (vLLM docs). Use fp8-w8a8.") in msgs
    sm86 = pf.Gpu(0, "RTX 3090", "8.6", 0, 24576, 24576)
    assert ("warn", "fp8-w8a8 on SM 8.6 runs as W8A16 weight-only (Marlin) [per vLLM docs, unverified]") in \
        [(f.level, f.message) for f in pf.check_eval_sm(sm86, ["fp8-w8a8"])]


def test_eval_sm_89_is_verified_and_quiet(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    assert pf.check_eval_sm(gpu, ["int4-w4a16", "int4-w4a8-fp8", "int4-w4a8-int8", "int8-w8a8", "fp8-w8a8"]) == []


def test_eval_fit_olmoe_bf16_skipped_quantized_fits(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    reason = pf.check_eval_fit("bf16 baseline", int(12.89 * pf.GIB), gpu)
    assert reason == "bf16 baseline does not fit: 13.8 GB weights > 13.5 GB budget (0.85 x 16.0 GiB - 1 GiB)"
    assert pf.check_eval_fit("int4-w4a16", int(3.7 * GB), gpu) is None
    assert pf.check_eval_fit("x", None, gpu) is None


def test_eval_free_vram(fixtures):
    assert pf.check_eval_free(gpus_of(fixtures, "rtx4070")[0]) == []
    busy = pf.Gpu(0, "x", "8.9", 8000, 16376, 8376)
    assert "vLLM needs 13.6 GiB free on GPU 0" in pf.check_eval_free(busy)[0].message


W4A16_QC = {"quant_method": "compressed-tensors", "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
                                                      "group_size": 128}}}}


def test_act_checks(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    assert pf.check_act(gpu, W4A16_QC) == []
    w8 = {"quant_method": "compressed-tensors",
          "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel"}}}}
    assert pf.check_act(gpu, w8)[0].message == \
        "--act needs an INT4 symmetric group-wise checkpoint, this model is compressed-tensors int8 symmetric=True strategy=channel"
    a100 = pf.Gpu(0, "A100", "8.0", 0, 40000, 40000)
    assert pf.check_act(a100, W4A16_QC)[0].message.startswith("--act needs SM 8.9 or 12.0")
    assert pf.is_int4_sym_group(None) == (False, "not quantized")


def test_run_lock(tmp_path):
    path = tmp_path / "q" / ".lock"
    lock = pf.RunLock(path)
    lock.acquire()
    assert path.read_text() == str(os.getpid())
    lock.release()
    assert not path.exists()
    path.write_text("1")  # PID 1 is always alive
    with pytest.raises(pf.PreflightError, match=r"another qtool run \(pid 1\) is active"):
        pf.RunLock(path).acquire()
    path.write_text("999999999")  # dead PID: stale lock is taken over
    pf.RunLock(path).acquire()
```

- [ ] **Step 2: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_preflight.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool.preflight'`

- [ ] **Step 3: Write the implementation** — `qtool/preflight.py`

```python
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
```

- [ ] **Step 4: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_preflight.py`
Expected: `18 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_preflight.py qtool/preflight.py
git commit -m "Add preflight: GPU selection, thresholds, SM and --act checks, run lock"
```

---

### Task 6: report.py — Δ pp, recovery %, bảng tổng hợp

Chỉ tính những gì spec cho phép tool tự tính. Baseline thiếu / bị bỏ → `n/a`, không bịa so sánh.

**Files:**
- Create: `qtool/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `parsers.Metric`, `console.fmt_gb`.
- Produces: `delta_pp`, `recovery`, `Row(label,status,metrics,size_bytes,note)`, `vs_baseline(m, base) -> (delta, recovery)`, `summary_lines(rows, task, baseline)`, `summary_title(task, rows, limit)`, `write_summary(path, rows, meta)`.

- [ ] **Step 1: Write the failing test** — `tests/test_report.py`

```python
# ABOUTME: Tests for qtool.report: Δ pp, recovery %, the summary table (spec §8 layout) and summary.json.
# ABOUTME: Baseline missing or skipped must show n/a instead of a made-up comparison.
import json

import pytest

from qtool import report
from qtool.parsers import Metric


def gsm(v, se=0.011):
    return {"gsm8k": Metric("gsm8k", "exact_match", "strict-match", v, se)}


def test_delta_and_recovery():
    assert report.delta_pp(0.792, 0.812) == pytest.approx(-2.0)
    assert report.recovery(0.792, 0.812) == pytest.approx(97.536, abs=1e-3)
    assert report.recovery(0.5, 0.0) is None


def test_summary_table_matches_spec_layout():
    rows = [report.Row("bf16", "OK", gsm(0.812), 8_800_000_000),
            report.Row("int4-w4a16", "OK", gsm(0.792), 5_200_000_000),
            report.Row("int4-w4a8-fp8", "OK", gsm(0.784), 5_200_000_000, "shared"),
            report.Row("int8-w8a8", "FAILED")]
    assert report.summary_title("gsm8k", rows, None) == "Summary · gsm8k strict-match (full)"
    assert report.summary_lines(rows, "gsm8k", "bf16") == [
        "  scheme          status   accuracy        Δ vs base   recovery   size",
        "  bf16            OK       0.812 ± 0.011   —           —          8.8 GB",
        "  int4-w4a16      OK       0.792 ± 0.011   −2.0 pp     97.5%      5.2 GB",
        "  int4-w4a8-fp8   OK       0.784 ± 0.011   −2.8 pp     96.6%      5.2 GB (shared)",
        "  int8-w8a8       FAILED   —               n/a         n/a        unknown",
    ]


def test_skipped_baseline_gives_na():
    rows = [report.Row("bf16", "SKIPPED", note="does not fit"), report.Row("int4-w4a16", "OK", gsm(0.68, 0.067))]
    assert report.summary_lines(rows, "gsm8k", "bf16")[2] == \
        "  int4-w4a16      OK       0.680 ± 0.067   n/a         n/a        unknown"
    assert report.summary_title("gsm8k", rows, 50) == "Summary · gsm8k strict-match (limit 50)"


def test_long_labels_widen_the_first_column():
    rows = [report.Row("baseline", "OK", gsm(0.70, 0.105), 9_300_000_000),
            report.Row("INT4-W4A16+act-fp8", "OK", gsm(0.75, 0.099), 5_500_000_000)]
    assert report.summary_lines(rows, "gsm8k", "baseline") == [
        "  scheme              status   accuracy        Δ vs base   recovery   size",
        "  baseline            OK       0.700 ± 0.105   —           —          9.3 GB",
        "  INT4-W4A16+act-fp8  OK       0.750 ± 0.099   +5.0 pp     107.1%     5.5 GB",
    ]


def test_write_summary(tmp_path):
    path = tmp_path / "results" / "summary.json"
    report.write_summary(path, [report.Row("bf16", "OK", gsm(0.8), 1)], {"model": "m"})
    data = json.loads(path.read_text())
    assert data["meta"] == {"model": "m"}
    assert data["rows"][0]["metrics"]["gsm8k"]["value"] == 0.8 and data["rows"][0]["status"] == "OK"
```

- [ ] **Step 2: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_report.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool.report'`

- [ ] **Step 3: Write the implementation** — `qtool/report.py`

```python
# ABOUTME: Summary of a run: Δ pp and recovery % against the baseline (only numbers qtool computes itself),
# ABOUTME: the summary table printed at the end and summary.json written next to the results.
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from qtool.console import fmt_gb
from qtool.parsers import Metric


def delta_pp(value: float, base: float) -> float:
    return (value - base) * 100


def recovery(value: float, base: float) -> float | None:
    return value / base * 100 if base > 0 else None


@dataclass
class Row:
    label: str
    status: str  # OK | FAILED | SKIPPED
    metrics: dict[str, Metric] = field(default_factory=dict)  # primary metric per task
    size_bytes: int | None = None
    note: str = ""


def _score(m: Metric | None) -> str:
    if m is None:
        return "—"
    return f"{m.value:.3f} ± {m.stderr:.3f}" if m.stderr is not None else f"{m.value:.3f}"


def vs_baseline(m: Metric | None, base: Metric | None) -> tuple[str, str]:
    if m is None or base is None:
        return "n/a", "n/a"
    rec = recovery(m.value, base.value)
    return f"{delta_pp(m.value, base.value):+.1f} pp".replace("-", "−"), "n/a" if rec is None else f"{rec:.1f}%"


def summary_lines(rows: list[Row], task: str, baseline: str | None) -> list[str]:
    base_row = next((r for r in rows if r.label == baseline), None)
    base = base_row.metrics.get(task) if base_row else None
    w = max([16] + [len(r.label) + 2 for r in rows])
    out = [f"  {'scheme':<{w}}{'status':<9}{'accuracy':<16}{'Δ vs base':<12}{'recovery':<11}size"]
    for r in rows:
        m = r.metrics.get(task)
        delta, rec = ("—", "—") if r.label == baseline else vs_baseline(m, base)
        size = fmt_gb(r.size_bytes) + (f" ({r.note})" if r.note else "")
        out.append(f"  {r.label:<{w}}{r.status:<9}{_score(m):<16}{delta:<12}{rec:<11}{size}")
    return out


def summary_title(task: str, rows: list[Row], limit: int | None) -> str:
    metric = next((r.metrics[task] for r in rows if task in r.metrics), None)
    what = f"{task} {metric.filter}" if metric and metric.filter != "none" else task
    return f"Summary · {what} ({'full' if limit is None else f'limit {limit}'})"


def write_summary(path: Path, rows: list[Row], meta: dict) -> None:
    data = {"meta": meta, "rows": [{**asdict(r), "metrics": {t: asdict(m) for t, m in r.metrics.items()}} for r in rows]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
```

- [ ] **Step 4: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_report.py`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_report.py qtool/report.py
git commit -m "Add report: delta pp, recovery and summary table"
```

---

### Task 7: plan.py — runs dir, cache checkpoint / baseline, kế hoạch

Checkpoint dùng lại khi `qtool.json` → `signature` khớp (model, checkpoint, recipe, ignore, calib, samples, seq len, llmcompressor); lệch → `stale` (exit 2, gợi ý `--force`). Baseline dùng lại theo từng task khi khớp model / limit / image / mode / GPU.

**Files:**
- Create: `qtool/plan.py`
- Test: `tests/test_plan.py`

**Interfaces:**
- Consumes: `schemes.CHECKPOINTS/SCHEMES`.
- Produces: `model_key`, `runs_dir(output_dir, model)`, `checkpoint_signature(...)`, `calib_key(...)`, `diff(saved, wanted, names)`, `CheckpointState(name,schemes,action,diff)` (action `build|rebuild|reuse|stale`), `checkpoint_states(rd, needed, wanted, force)`, `eval_meta(model,task,limit,image,mode,gpu)`, `result_path(rd,label,task)`, `cached_result(path, wanted) -> (valid, reason)`, `render_plan(...)`.

- [ ] **Step 1: Write the failing test** — `tests/test_plan.py`

```python
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


def test_cached_result(tmp_path):
    p = tmp_path / "bf16_gsm8k.json"
    wanted = plan.eval_meta("Qwen/Qwen3.5-4B", "gsm8k", None, "qtool-eval:abc", "chat", "RTX 4070")
    assert plan.cached_result(p, wanted) == (False, "cache: none, will run")
    p.write_text(json.dumps({"qtool": {**wanted, "limit": 250}}))
    assert plan.cached_result(p, wanted) == (False, "cache stale: eval-limit: 250 → full, will run")
    p.write_text(json.dumps({"qtool": wanted}))
    assert plan.cached_result(p, wanted) == (True, "cache: valid, reused")


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
```

- [ ] **Step 2: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_plan.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool.plan'`

- [ ] **Step 3: Write the implementation** — `qtool/plan.py`

```python
# ABOUTME: Run planning: where a model's runs live, which checkpoints to build or reuse (qtool.json signature),
# ABOUTME: whether a cached baseline result is still valid, and the plan text printed before anything runs.
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from qtool import schemes


def model_key(model: str) -> str:
    return Path(model.rstrip("/")).name


def runs_dir(output_dir: Path, model: str) -> Path:
    p = Path(model).expanduser()
    if p.is_dir() and p.parent.name == "checkpoints":  # a checkpoint qtool made: keep results next to it
        return p.parent.parent
    return Path(output_dir).expanduser() / model_key(model)


def checkpoint_signature(model: str, ckpt: str, calib: str, samples: int, seq_len: int, ignore: list[str],
                         llmcompressor: str) -> dict:
    return {"model": model, "checkpoint": ckpt, "recipe": schemes.CHECKPOINTS[ckpt]["recipe"], "ignore": sorted(ignore),
            "calib": calib, "calib_samples": samples, "calib_seq_len": seq_len, "llmcompressor": llmcompressor}


def calib_key(model: str, calib: str, samples: int, seq_len: int) -> str:
    return hashlib.sha1(json.dumps([model, calib, samples, seq_len]).encode()).hexdigest()[:12]


def _show(v) -> str:
    return "full" if v is None else str(v)


def diff(saved: dict, wanted: dict, names: dict | None = None) -> list[str]:
    names = names or {}
    return [f"{names.get(k, k)}: {_show(saved.get(k))} → {_show(wanted[k])}" for k in wanted if saved.get(k) != wanted[k]]


@dataclass
class CheckpointState:
    name: str
    schemes: list[str]
    action: str  # build | rebuild | reuse | stale
    diff: list[str]


def checkpoint_states(rd: Path, needed: dict[str, list[str]], wanted: dict[str, dict], force: bool) -> list[CheckpointState]:
    out = []
    for name, users in needed.items():
        meta = rd / "checkpoints" / name / "qtool.json"
        if not meta.exists():
            out.append(CheckpointState(name, users, "build", []))
        elif force:
            out.append(CheckpointState(name, users, "rebuild", []))
        else:
            d = diff(json.loads(meta.read_text()).get("signature", {}), wanted[name])
            out.append(CheckpointState(name, users, "stale" if d else "reuse", d))
    return out


EVAL_FIELDS = {"model": "model", "limit": "eval-limit", "image": "eval image", "mode": "eval mode", "gpu": "GPU"}


def eval_meta(model: str, task: str, limit: int | None, image: str, mode: str, gpu: str) -> dict:
    return {"model": model, "task": task, "limit": limit, "image": image, "mode": mode, "gpu": gpu}


def result_path(rd: Path, label: str, task: str) -> Path:
    return rd / "results" / f"{label}_{task}.json"


def cached_result(path: Path, wanted: dict) -> tuple[bool, str]:
    """(valid, reason) for a cached per-task result file written by flow.save_task_results."""
    if not path.exists():
        return False, "cache: none, will run"
    d = diff(json.loads(path.read_text()).get("qtool", {}), wanted, EVAL_FIELDS)
    if d:
        return False, "cache stale: " + "; ".join(d) + ", will run"
    return True, "cache: valid, reused"


def render_plan(model: str, states: list[CheckpointState], schemes_: list[str], tasks: list[str], limit: int | None,
                baseline_note: str | None, no_eval: bool) -> list[str]:
    out = [f"Plan for {model} ({len(schemes_)} scheme{'s' if len(schemes_) != 1 else ''})"]
    for st in states:
        users = ", ".join(st.schemes)
        if st.action in ("build", "rebuild"):
            out.append(f"  quantize  {st.name:<11} -> {'shared by' if len(st.schemes) > 1 else 'used by'}: {users}"
                       + ("  (--force)" if st.action == "rebuild" else ""))
        elif st.action == "reuse":
            out.append(f"  reuse     {st.name:<11} -> checkpoints/{st.name} (qtool.json matches)")
        else:
            out.append(f"  stale     {st.name:<11} -> checkpoints/{st.name} differs: " + "; ".join(st.diff))
    for s in schemes_:
        if schemes.SCHEMES[s].quantization:
            out.append(f"  online    {s:<11} -> vLLM --quantization {schemes.SCHEMES[s].quantization} (no checkpoint)")
    if no_eval:
        out.append("  eval      skipped (--no-eval)")
    else:
        out.append(f"  eval      bf16 baseline ({baseline_note})")
        out.append(f"  eval      {', '.join(schemes_)}   tasks: {','.join(tasks)} ({_show(limit)})")
    return out
```

- [ ] **Step 4: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_plan.py`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_plan.py qtool/plan.py
git commit -m "Add run planning: checkpoint reuse, baseline cache, plan text"
```

---

### Task 8: images.py + Dockerfile + script test container

Tag image = hash Dockerfile (sửa Dockerfile → tag mới → tự build lại). Thêm layer `pytest` (compress, cho test container) và `num2words` (eval, spike Q9) **ở dòng `RUN` riêng sau các dòng cũ** để Docker dùng lại cache layer torch / lm-eval đã build trong spike.

**Files:**
- Create: `qtool/images.py`
- Create: `scripts/test-container.sh`
- Test: `tests/test_images.py`

**Interfaces:**
- Produces: `REPO`, `IMAGES`, `SCRIPTS` (= `images/scripts`), `dockerfile(kind)`, `compress_cuda(cc) -> 'cu126'|'cu129'`, `tag(kind, cuda=None)`, `pinned(kind, package)`, `build_argv(kind, cuda=None)`, `exists(image, run)`.

- [ ] **Step 1: Update the compress Dockerfile** — `images/compress/Dockerfile`

```dockerfile
# ABOUTME: Image cho step 1-3 cua qtool: transformers + llmcompressor tren torch 2.13.
# ABOUTME: CUDA=cu129 cho SM 7.5-12.0, CUDA=cu126 cho SM 5.0-9.0 (co V100); ghim theo spike 2026-10-30.
FROM python:3.11-slim
ARG CUDA=cu129
# Triton can C compiler de build helper luc chay (kernel Hessian GPTQ)
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/${CUDA} && \
    pip install --no-cache-dir llmcompressor==0.13.0 transformers==5.14.1 datasets
# Container-tier tests (scripts/test-container.sh) run inside this image
RUN pip install --no-cache-dir pytest==9.2.0
```

- [ ] **Step 2: Update the eval Dockerfile** — `images/eval/Dockerfile`

```dockerfile
# ABOUTME: Image cho step 4 cua qtool: vLLM 0.28 (giong bake-off) + lm-eval.
# ABOUTME: Ghim torch/vllm/transformers/compressed-tensors cua image goc; huggingface-hub<2 cho transformers 5.15 (spike 2026-10-30).
FROM vllm/vllm-openai:v0.28.0-cu129-ubuntu2404
RUN pip freeze | grep -iE "^(torch|vllm|transformers|compressed-tensors)==" > /tmp/vllm-constraints.txt && \
    echo "huggingface-hub<2.0" >> /tmp/vllm-constraints.txt && \
    pip install --no-cache-dir -c /tmp/vllm-constraints.txt "lm-eval[api]==0.4.13"
# Processor SmolVLM / Idefics can num2words ca luc vLLM load model (spike Q9)
RUN pip install --no-cache-dir -c /tmp/vllm-constraints.txt num2words
ENTRYPOINT []
```

- [ ] **Step 3: Write the container test runner (then `chmod +x`)** — `scripts/test-container.sh`

```bash
#!/usr/bin/env bash
# ABOUTME: Runs the container-tier tests (tests/container) inside the qtool-compress image on CPU.
# ABOUTME: Builds the image first if missing; usage: scripts/test-container.sh [test paths, default tests/container]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$(python3 -c 'from qtool import images; print(images.tag("compress", "cu129"))')
docker image inspect "$TAG" >/dev/null 2>&1 || docker build -t "$TAG" --build-arg CUDA=cu129 images/compress
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 -v "$PWD:/src:ro" -w /src \
  -e PYTHONDONTWRITEBYTECODE=1 "$TAG" python -m pytest -q -m container -p no:cacheprovider "${@:-tests/container}"
```

- [ ] **Step 4: Write the failing test** — `tests/test_images.py`

```python
# ABOUTME: Tests for qtool.images: hash-based tags, cu126/cu129 choice by SM, versions pinned in the Dockerfiles
# ABOUTME: (eval image must carry num2words, compress image pytest) and the docker build command.
import re

import pytest

from qtool import images


def test_tags_follow_dockerfile_hash():
    assert re.fullmatch(r"qtool-compress:cu129-[0-9a-f]{12}", images.tag("compress", "cu129"))
    assert re.fullmatch(r"qtool-eval:[0-9a-f]{12}", images.tag("eval"))
    assert images.tag("compress", "cu126").split("-")[-1] == images.tag("compress", "cu129").split("-")[-1]


@pytest.mark.parametrize("cc,cuda", [("7.0", "cu126"), ("6.1", "cu126"), ("7.5", "cu129"), ("8.9", "cu129"), ("12.0", "cu129")])
def test_compress_cuda(cc, cuda):
    assert images.compress_cuda(cc) == cuda


def test_pins():
    assert images.pinned("compress", "llmcompressor") == "0.13.0"
    assert images.pinned("compress", "transformers") == "5.14.1"
    assert images.pinned("eval", "lm-eval[api]") == "0.4.13"
    assert "num2words" in images.dockerfile("eval").read_text()
    assert images.pinned("compress", "nope") is None


def test_build_argv():
    argv = images.build_argv("compress", "cu126")
    assert argv[:3] == ["docker", "build", "-t"] and argv[3] == images.tag("compress", "cu126")
    assert argv[4:6] == ["--build-arg", "CUDA=cu126"] and argv[-1].endswith("images/compress")
    assert images.build_argv("eval")[4].endswith("images/eval")


def test_exists_uses_docker_inspect():
    calls = []

    class R:
        returncode = 0

    assert images.exists("x:1", run=lambda argv, **kw: calls.append(argv) or R()) is True
    assert calls == [["docker", "image", "inspect", "x:1"]]
```

- [ ] **Step 5: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_images.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool.images'`

- [ ] **Step 6: Write the implementation** — `qtool/images.py`

```python
# ABOUTME: Docker images qtool needs: tags derived from the Dockerfile hash (edit Dockerfile = new tag = rebuild),
# ABOUTME: compress variant by SM (cu126 has sm_70, cu129 has sm_75..sm_120), pinned versions and build commands.
import hashlib
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "images"
SCRIPTS = IMAGES / "scripts"


def dockerfile(kind: str) -> Path:
    return IMAGES / kind / "Dockerfile"


def _hash(kind: str) -> str:
    return hashlib.sha256(dockerfile(kind).read_bytes()).hexdigest()[:12]


def compress_cuda(cc: str) -> str:
    major, minor = (int(x) for x in cc.split("."))
    return "cu126" if (major, minor) < (7, 5) else "cu129"


def tag(kind: str, cuda: str | None = None) -> str:
    if kind == "compress":
        return f"qtool-compress:{cuda}-{_hash(kind)}"
    return f"qtool-{kind}:{_hash(kind)}"


def pinned(kind: str, package: str) -> str | None:
    m = re.search(rf"{re.escape(package)}==([\w.]+)", dockerfile(kind).read_text())
    return m[1] if m else None


def build_argv(kind: str, cuda: str | None = None) -> list[str]:
    args = ["--build-arg", f"CUDA={cuda}"] if kind == "compress" else []
    return ["docker", "build", "-t", tag(kind, cuda), *args, str(IMAGES / kind)]


def exists(image: str, run=subprocess.run) -> bool:
    return run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
```

- [ ] **Step 7: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_images.py`
Expected: `9 passed`

- [ ] **Step 8: Commit**

```bash
git add Update the compress Dockerfile Update the eval Dockerfile Write the container test runner (then `chmod +x`) tests/test_images.py qtool/images.py
git commit -m "Add image tags from Dockerfile hash; pytest and num2words layers"
```

---

### Task 9: images/scripts/recipe.py — recipe chung (spike Q9)

Suy recipe từ `config.json` + cấu trúc module cho mọi kiến trúc: loader, ignore (lm_head, vision/projector, router MoE, mixer hybrid, `--ignore`), layer tuần tự (`ModuleList` lớn nhất có class trong `_no_split_modules`), MTP (khoá `mtp`/`nextn` > 0), remote code, kiểm chat template (calib và eval). Test trên model tí hon ngẫu nhiên của 4 kiến trúc đã kiểm chứng; số kỳ vọng khớp spike Q9 thu nhỏ.

**Files:**
- Create: `images/scripts/recipe.py`
- Create: `tests/container/tinyhelpers.py`
- Create: `tests/container/conftest.py`
- Test: `tests/container/test_recipe.py`

**Interfaces:**
- Produces (dict config, torch lazy): `text_cfg`, `orig_dtype`, `is_multimodal`, `loader_name`, `num_experts`, `is_hybrid`, `mtp_present`, `remote_code_required(cfg, transformers_module)`, `vision_segment`, `derive(model, cfg, extra_ignore) -> {ignore, why, sequential_target, num_layers, num_experts, mtp, linears_total, linears_quantized, quantized_sample}`, `keeps_content`, `render_calib(tok, conversations) -> (mode, texts)`, `eval_chat_mode(tok) -> (mode, reason)`.
- Test helpers: `tinyhelpers.{CORPUS, CHAT_TEMPLATE, PARTS_ONLY_TEMPLATE, make_tokenizer, build_model, events}`, fixtures `tiny_dir(name, chat_template)`, `calib_jsonl`.

Test chạy **trong image compress trên rtx4070 (CPU)**: `./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh <file>'`. Lần đầu script build image `qtool-compress:cu129-<hash>` (chỉ thêm layer pytest nếu cache spike còn).

- [ ] **Step 1: Write the tiny-model helpers** — `tests/container/tinyhelpers.py`

```python
# ABOUTME: Helpers for container-tier tests: tiny random models from shrunk real configs of the 4 verified
# ABOUTME: architectures, a tiny trained tokenizer, chat templates and @qtool event extraction.
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "tests" / "fixtures" / "tiny-configs"
sys.path.insert(0, str(ROOT / "images" / "scripts"))

CORPUS = [f"calibration sentence number {i} about apples rivers and small engines" for i in range(64)]
CHAT_TEMPLATE = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}"
# Mimics SmolVLM: renders only list-of-parts content, silently drops string content
PARTS_ONLY_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{% if m['content'] is not string %}"
                       "{% for p in m['content'] %}{{ p['text'] }}{% endfor %}{% endif %}</s>{% endfor %}")


def make_tokenizer(chat_template=CHAT_TEMPLATE):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<s>", "</s>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                  pad_token="</s>")
    tok.chat_template = chat_template
    return tok


def build_model(name: str, dtype: str = "bfloat16"):
    import torch
    import transformers
    from recipe import loader_name

    d = json.loads((TINY / f"{name}.json").read_text())
    cfg = transformers.AutoConfig.for_model(**d)
    torch.manual_seed(0)
    return getattr(transformers, loader_name(d)).from_config(cfg, dtype=getattr(torch, dtype)), d




def events(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        i = line.find("@qtool ")
        if i >= 0:
            out.append(json.loads(line[i + 7:]))
    return out
```

- [ ] **Step 2: Write the container fixtures** — `tests/container/conftest.py`

```python
# ABOUTME: Fixtures for container-tier tests (run inside qtool-compress, CPU only): saved tiny model dirs
# ABOUTME: and a calibration .jsonl built from the helpers in tinyhelpers.py.
import json
from pathlib import Path

import pytest

from tinyhelpers import CHAT_TEMPLATE, CORPUS, build_model, make_tokenizer


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    """Factory: tiny_dir(name, chat_template=...) -> saved model dir (weights stored as bfloat16)."""
    made = {}

    def make(name: str, chat_template=CHAT_TEMPLATE) -> Path:
        key = (name, chat_template)
        if key not in made:
            out = tmp_path_factory.mktemp(f"src-{name}")
            model, _ = build_model(name)
            model.save_pretrained(out)
            make_tokenizer(chat_template).save_pretrained(out)
            made[key] = out
        return made[key]

    return make


@pytest.fixture(scope="session")
def calib_jsonl(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("calib") / "calib.jsonl"
    with open(path, "w") as f:
        for i, text in enumerate(CORPUS):
            f.write(json.dumps({"messages": [{"role": "user", "content": text},
                                             {"role": "assistant", "content": f"answer {i} about engines"}]}) + "\n")
    return path
```

- [ ] **Step 3: Write the failing test** — `tests/container/test_recipe.py`

```python
# ABOUTME: Container-tier tests for recipe.derive on tiny random models of each verified architecture.
# ABOUTME: Expected ignore lists / layer classes mirror what spike Q9 derived on the full-size models.
import pytest

from tinyhelpers import PARTS_ONLY_TEMPLATE, build_model, make_tokenizer

pytestmark = pytest.mark.container


def derive_for(name, extra=()):
    from llmcompressor.modeling.moe.linearize import linearize_moe
    from recipe import derive

    model, cfg = build_model(name)
    linearize_moe(model)
    return derive(model, cfg, extra)


def test_qwen3_5_hybrid_multimodal_mtp():
    r = derive_for("qwen3_5")
    assert r["ignore"] == ["re:.*lm_head", "re:.*visual.*", "re:.*linear_attn.*"]
    assert r["sequential_target"] == "Qwen3_5DecoderLayer" and r["num_layers"] == 4
    assert r["mtp"] is True
    assert (r["linears_total"], r["linears_quantized"]) == (38, 16)


def test_phi3_dense():
    r = derive_for("phi3")
    assert r["ignore"] == ["re:.*lm_head"]
    assert r["sequential_target"] == "Phi3DecoderLayer"
    assert (r["linears_total"], r["linears_quantized"]) == (9, 8)
    assert r["mtp"] is False


def test_olmoe_moe_is_linearized_before_derive():
    r = derive_for("olmoe")
    assert r["num_experts"] == 8
    assert r["sequential_target"] == "OlmoeDecoderLayer"  # not the 8-expert ModuleList
    assert r["ignore"] == ["re:.*lm_head"]  # router is OlmoeTopKRouter, not a Linear
    assert (r["linears_total"], r["linears_quantized"]) == (57, 56)


def test_smolvlm_vision_and_connector_ignored():
    r = derive_for("smolvlm")
    assert r["ignore"] == ["re:.*lm_head", "re:.*connector.*", "re:.*vision_model.*"]
    assert r["sequential_target"] == "LlamaDecoderLayer" and r["num_layers"] == 2
    assert (r["linears_total"], r["linears_quantized"]) == (22, 14)


def test_extra_ignore_is_added_not_replacing():
    r = derive_for("phi3", ["re:.*down_proj"])
    assert r["ignore"] == ["re:.*lm_head", "re:.*down_proj"]
    assert r["why"]["re:.*down_proj"] == "--ignore"
    assert r["linears_quantized"] == 6


def test_render_calib_falls_back_to_parts_then_raw():
    from recipe import render_calib

    convs = [[{"role": "user", "content": "rivers and engines"}]]
    assert render_calib(make_tokenizer(), convs)[0] == "chat_template(str)"
    assert render_calib(make_tokenizer(PARTS_ONLY_TEMPLATE), convs)[0] == "chat_template(parts)"
    assert render_calib(make_tokenizer(None), convs) == ("raw", ["rivers and engines"])


def test_eval_chat_mode():
    from recipe import eval_chat_mode

    assert eval_chat_mode(make_tokenizer())[0] == "chat"
    assert eval_chat_mode(make_tokenizer(PARTS_ONLY_TEMPLATE)) == ("completion", "chat template drops string message content")
    assert eval_chat_mode(make_tokenizer(None)) == ("completion", "tokenizer has no chat template")
```

- [ ] **Step 4: Run the test to see it fail**

Run: `./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh tests/container/test_recipe.py'`
Expected: FAIL — `ModuleNotFoundError: No module named 'recipe'`

- [ ] **Step 5: Write the implementation** — `images/scripts/recipe.py`

```python
# ABOUTME: Generic recipe derivation for any transformers model (spike Q9): loader class, ignore list,
# ABOUTME: sequential layer, MTP, remote-code and chat-template checks. Config helpers take plain dicts; torch is lazy.
import re

VISION_PREFIXES = ("visual", "vision_tower", "vision_model", "vision_encoder", "multi_modal_projector",
                   "mm_projector", "connector", "image_newline")
MIXERS = ("linear_attn", "mamba", "mixer", "ssm")
ATTENTION_TYPES = ("full_attention", "attention", "sliding_attention")
EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


def text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def orig_dtype(cfg: dict) -> str | None:
    for c in (cfg, text_cfg(cfg)):
        d = c.get("dtype") or c.get("torch_dtype")
        if d:
            return str(d).replace("torch.", "")
    return None


def is_multimodal(cfg: dict) -> bool:
    return "vision_config" in cfg


def loader_name(cfg: dict) -> str:
    return "AutoModelForImageTextToText" if is_multimodal(cfg) else "AutoModelForCausalLM"


def num_experts(cfg: dict) -> int | None:
    for k in EXPERT_KEYS:
        v = text_cfg(cfg).get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 1:
            return v
    return None


def is_hybrid(cfg: dict) -> bool:
    return any(t not in ATTENTION_TYPES for t in text_cfg(cfg).get("layer_types") or [])


def mtp_present(cfg: dict) -> bool:
    return any(("mtp" in k or "nextn" in k) and isinstance(v, int) and not isinstance(v, bool) and v > 0
               for k, v in text_cfg(cfg).items())


def remote_code_required(cfg: dict, transformers_module) -> bool:
    archs = cfg.get("architectures") or []
    return "auto_map" in cfg and not any(hasattr(transformers_module, a) for a in archs)


def vision_segment(name: str) -> str | None:
    for seg in name.split("."):
        if seg.startswith(VISION_PREFIXES):
            return seg
    return None


def _ignored(name: str, patterns: list[str]) -> bool:
    return any(re.match(p[3:], name) if p.startswith("re:") else name == p for p in patterns)


def derive(model, cfg: dict, extra_ignore=()) -> dict:
    import torch

    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    ignore, why = ["re:.*lm_head"], {"re:.*lm_head": "output head"}

    for seg in sorted({s for n, _ in model.named_modules() if (s := vision_segment(n))}):
        rx = f"re:.*{re.escape(seg)}.*"
        ignore.append(rx)
        why[rx] = "vision/projector"

    ne = num_experts(cfg)
    if ne:
        for suffix in sorted({re.sub(r"^.*?layers\.\d+\.", "", n) for n, m in linears
                              if m.out_features in (ne, 1) and not vision_segment(n)}):
            rx = f"re:.*{re.escape(suffix)}$"
            ignore.append(rx)
            why[rx] = f"MoE router (out_features in {{{ne}, 1}})"

    if is_hybrid(cfg):
        for p in MIXERS:
            if any(f".{p}." in n for n, _ in linears):
                rx = f"re:.*{p}.*"
                ignore.append(rx)
                why[rx] = "hybrid linear-attention / SSM mixer"

    for p in extra_ignore:
        if p not in ignore:
            ignore.append(p)
            why[p] = "--ignore"

    # Largest non-vision ModuleList whose element class is in _no_split_modules; size alone picks MoE expert lists
    no_split = set(getattr(model, "_no_split_modules", None) or [])
    lists = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) > 0 and not vision_segment(n)]
    preferred = [x for x in lists if type(x[1][0]).__name__ in no_split] or lists
    best = max(preferred, key=lambda x: len(x[1]), default=None)

    quantized = [n for n, _ in linears if not _ignored(n, ignore)]
    return {
        "ignore": ignore,
        "why": why,
        "sequential_target": type(best[1][0]).__name__ if best else None,
        "num_layers": len(best[1]) if best else None,
        "num_experts": ne,
        "mtp": mtp_present(cfg),
        "linears_total": len(linears),
        "linears_quantized": len(quantized),
        "quantized_sample": sorted({re.sub(r"\.\d+\.", ".N.", n) for n in quantized})[:12],
    }


def keeps_content(rendered: str, messages: list[dict]) -> bool:
    # Some templates (SmolVLM) silently drop string content; every message text must survive rendering
    return all(m["content"][:40] in rendered for m in messages if isinstance(m.get("content"), str) and m["content"])


def _as_parts(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]


def render_calib(tok, conversations: list[list[dict]]) -> tuple[str, list[str]]:
    """Return (mode, texts): 'chat_template(str)', 'chat_template(parts)' or 'raw' (template missing or drops text)."""
    if getattr(tok, "chat_template", None):
        as_str = [tok.apply_chat_template(c, tokenize=False) for c in conversations]
        if all(keeps_content(r, c) for r, c in zip(as_str[:8], conversations[:8])):
            return "chat_template(str)", as_str
        as_parts = [tok.apply_chat_template(_as_parts(c), tokenize=False) for c in conversations]
        if all(keeps_content(r, c) for r, c in zip(as_parts[:8], conversations[:8])):
            return "chat_template(parts)", as_parts
    return "raw", ["\n\n".join(m["content"] for m in c) for c in conversations]


def eval_chat_mode(tok) -> tuple[str, str]:
    """lm_eval sends string content; chat mode only if the template keeps it. Returns (mode, reason)."""
    sample = [{"role": "user", "content": "Probe question: what is two plus three?"},
              {"role": "assistant", "content": "Probe answer: two plus three is five."}]
    if not getattr(tok, "chat_template", None):
        return "completion", "tokenizer has no chat template"
    if not keeps_content(tok.apply_chat_template(sample, tokenize=False), sample):
        return "completion", "chat template drops string message content"
    return "chat", "chat template keeps message content"
```

- [ ] **Step 6: Run the test to see it pass**

Run: `./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh tests/container/test_recipe.py'`
Expected: `7 passed`

- [ ] **Step 7: Commit**

```bash
git add Write the tiny-model helpers Write the container fixtures tests/container/test_recipe.py images/scripts/recipe.py
git commit -m "Add generic recipe derivation with container-tier tests"
```

---

### Task 10: images/scripts/quantize.py — step 1–3 trong container

Một container = một checkpoint. In `@qtool` cho mọi sự kiện (step, config, load, linearize, recipe, calib, oneshot, saved, refused). Chặn model đã quantize / cần remote code **trước khi load weight** (exit 3). Tập calib token hoá lưu ở `--calib-cache` và dùng lại. Compute FP32 → cast về dtype gốc trước khi lưu (spike Q6). Copy file phụ thay vì khởi tạo processor (num2words). Ghi `qtool.json` (signature host truyền vào, recipe, calib, phiên bản).

**Files:**
- Create: `images/scripts/quantize.py`
- Test: `tests/container/test_quantize.py`

**Interfaces:**
- Consumes: `recipe.*`.
- Produces: CLI `python3 /qtool/quantize.py --model --checkpoint INT4-W4A16|INT8-W8A8 --compute-dtype --calib --samples --seq-len --calib-cache --ignore --signature --out`; exit 0 / 3 (refused) / khác (lỗi thư viện); `qtool.json` = `{signature, recipe, calib, compute_dtype, cast, oneshot_seconds, versions}`.

Test chạy **trong image compress trên rtx4070 (CPU)**: `./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh <file>'`. Lần đầu script build image `qtool-compress:cu129-<hash>` (chỉ thêm layer pytest nếu cache spike còn).

- [ ] **Step 1: Write the failing test** — `tests/container/test_quantize.py`

```python
# ABOUTME: Container-tier end-to-end tests for quantize.py on tiny models (CPU): events, checkpoint format,
# ABOUTME: float32-compute cast back to the source dtype, calibration cache, refusal of already-quantized models.
import json

import pytest

from tinyhelpers import events

pytestmark = pytest.mark.container


def run(capsys, src, out, calib, cache, checkpoint="INT4-W4A16", dtype="bfloat16", extra=()):
    import quantize

    code = quantize.main(["--model", str(src), "--checkpoint", checkpoint, "--compute-dtype", dtype,
                          "--calib", str(calib), "--samples", "16", "--seq-len", "64", "--calib-cache", str(cache),
                          "--signature", json.dumps({"model": str(src), "checkpoint": checkpoint}),
                          "--out", str(out), *extra])
    return code, events(capsys.readouterr().out)


def by_name(evs):
    return {e["event"]: e for e in evs}


def test_w4a16_phi3_float32_compute_casts_back(capsys, tiny_dir, calib_jsonl, tmp_path):
    out = tmp_path / "INT4-W4A16"
    code, evs = run(capsys, tiny_dir("phi3"), out, calib_jsonl, tmp_path / "calib", dtype="float32")
    assert code == 0
    assert [e["n"] for e in evs if e["event"] == "step"] == [1, 2, 3]
    e = by_name(evs)
    assert e["config"]["architectures"] == ["Phi3ForCausalLM"] and e["config"]["dtype"] == "bfloat16"
    assert e["recipe"]["ignore"] == ["re:.*lm_head"]
    assert e["calib"]["template"] == "chat_template(str)" and e["calib"]["n"] == 16 and e["calib"]["cached"] is False
    assert e["saved"]["cast"] == "float32->bfloat16" and e["saved"]["bytes"] > 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["dtype"] == "bfloat16" and cfg["quantization_config"]["format"] == "pack-quantized"
    meta = json.loads((out / "qtool.json").read_text())
    assert meta["signature"]["checkpoint"] == "INT4-W4A16" and meta["versions"]["llmcompressor"] == "0.13.0"


def test_calib_cache_is_reused(capsys, tiny_dir, calib_jsonl, tmp_path):
    cache = tmp_path / "calib"
    run(capsys, tiny_dir("phi3"), tmp_path / "a", calib_jsonl, cache)
    code, evs = run(capsys, tiny_dir("phi3"), tmp_path / "b", calib_jsonl, cache, checkpoint="INT8-W8A8")
    assert code == 0
    assert by_name(evs)["calib"]["cached"] is True
    cfg = json.loads((tmp_path / "b" / "config.json").read_text())
    assert cfg["quantization_config"]["format"] == "int-quantized"


def test_w8a8_olmoe_linearizes_experts(capsys, tiny_dir, calib_jsonl, tmp_path):
    code, evs = run(capsys, tiny_dir("olmoe"), tmp_path / "out", calib_jsonl, tmp_path / "calib", checkpoint="INT8-W8A8")
    assert code == 0
    e = by_name(evs)
    assert e["linearize"]["fused_expert_modules"] == 2
    assert e["recipe"]["linears_quantized"] == 56


def test_already_quantized_model_is_refused_before_loading(capsys, tiny_dir, calib_jsonl, tmp_path):
    first = tmp_path / "q"
    run(capsys, tiny_dir("phi3"), first, calib_jsonl, tmp_path / "calib")
    code, evs = run(capsys, first, tmp_path / "again", calib_jsonl, tmp_path / "calib")
    assert code == 3
    assert [e["event"] for e in evs] == ["step", "refused"]
    assert "already quantized" in evs[-1]["reason"]


def test_extra_ignore_reaches_recipe(capsys, tiny_dir, calib_jsonl, tmp_path):
    code, evs = run(capsys, tiny_dir("phi3"), tmp_path / "out", calib_jsonl, tmp_path / "calib",
                    extra=["--ignore", "re:.*down_proj"])
    assert code == 0
    assert by_name(evs)["recipe"]["ignore"] == ["re:.*lm_head", "re:.*down_proj"]
```

- [ ] **Step 2: Run the test to see it fail**

Run: `./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh tests/container/test_quantize.py'`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantize'`

- [ ] **Step 3: Write the implementation** — `images/scripts/quantize.py`

```python
# ABOUTME: Runs inside qtool-compress: steps 1-3 for one checkpoint (load model, calibration data, oneshot + save).
# ABOUTME: Prints `@qtool {json}` events for the host; library output (tqdm, loguru) passes through untouched.
import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipe import derive, is_multimodal, loader_name, orig_dtype, remote_code_required, render_calib  # noqa: E402

PRESETS = {"INT4-W4A16": "W4A16", "INT8-W8A8": "W8A8"}
EXIT_REFUSED = 3
SIDE_FILES = (".json", ".jinja", ".txt", ".model")


def emit(event: str, **fields) -> None:
    print("@qtool " + json.dumps({"event": event, **fields}, default=str), flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", choices=sorted(PRESETS), required=True)
    ap.add_argument("--compute-dtype", choices=["bfloat16", "float32"], required=True)
    ap.add_argument("--calib", required=True, help="HF dataset id[:split] or .jsonl file")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--calib-cache", required=True, help="directory for the tokenized calibration set")
    ap.add_argument("--ignore", default="", help="extra ignore patterns, comma separated")
    ap.add_argument("--signature", default="{}", help="JSON written verbatim into qtool.json for cache checks")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def load_rows(source: str, samples: int) -> list:
    from datasets import load_dataset

    if source.endswith(".jsonl"):
        ds = load_dataset("json", data_files=source, split="train")
    else:
        name, _, split = source.partition(":")
        ds = load_dataset(name, split=split or "train")
    ds = ds.shuffle(seed=42).select(range(min(samples, len(ds))))
    if "messages" in ds.column_names:
        return [list(m) for m in ds["messages"]]
    if "text" in ds.column_names:
        return list(ds["text"])
    raise ValueError(f"calibration source {source} has neither a 'messages' nor a 'text' column: {ds.column_names}")


def calibration(a, tok):
    from datasets import Dataset, load_from_disk

    cache = Path(a.calib_cache)
    if (cache / "qtool-calib.json").exists():
        stats = json.loads((cache / "qtool-calib.json").read_text())
        return load_from_disk(str(cache / "data")), {**stats, "cached": True, "seconds": 0.0}
    t0 = time.time()
    rows = load_rows(a.calib, a.samples)
    if rows and isinstance(rows[0], str):
        mode, texts = "text", rows
    else:
        mode, texts = render_calib(tok, rows)
    enc = tok(texts, padding=False, max_length=a.seq_len, truncation=True, add_special_tokens=mode in ("text", "raw"))
    ds = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    lens = [len(x) for x in enc["input_ids"]]
    stats = {"source": a.calib, "template": mode, "n": len(lens), "min": min(lens), "median": statistics.median(lens),
             "max": max(lens), "truncated": sum(1 for n in lens if n >= a.seq_len)}
    tmp = cache.with_name(".tmp-" + cache.name)
    shutil.rmtree(tmp, ignore_errors=True)
    ds.save_to_disk(str(tmp / "data"))
    (tmp / "qtool-calib.json").write_text(json.dumps(stats))
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(cache)
    return ds, {**stats, "cached": False, "seconds": round(time.time() - t0, 1)}


def copy_side_files(model: str, out: Path) -> list[str]:
    # Copy processor/template configs as files instead of instantiating a processor (pulls optional deps like num2words)
    if os.path.isdir(model):
        src = model
    else:
        from huggingface_hub import snapshot_download
        src = snapshot_download(model, allow_patterns=[f"*{ext}" for ext in SIDE_FILES])
    copied = []
    for f in sorted(os.listdir(src)):
        if f.endswith(SIDE_FILES) and not f.endswith(".index.json") and not (out / f).exists():
            shutil.copy(os.path.join(src, f), out / f)
            copied.append(f)
    return copied


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv=None) -> int:
    a = parse_args(argv)
    import torch
    import transformers

    emit("step", n=1)
    cfg, _ = transformers.PretrainedConfig.get_config_dict(a.model)
    if "quantization_config" in cfg:
        method = cfg["quantization_config"].get("quant_method", "unknown")
        emit("refused", reason=f"model is already quantized ({method})", hint="evaluate it with `qtool eval`")
        return EXIT_REFUSED
    if remote_code_required(cfg, transformers):
        emit("refused", reason="model needs trust_remote_code, which qtool does not support", hint="")
        return EXIT_REFUSED
    loader = getattr(transformers, loader_name(cfg))
    emit("config", architectures=cfg.get("architectures"), model_type=cfg.get("model_type"), dtype=orig_dtype(cfg),
         multimodal=is_multimodal(cfg), loader=loader.__name__)
    t0 = time.time()
    model = loader.from_pretrained(a.model, dtype=getattr(torch, a.compute_dtype))
    emit("load", seconds=round(time.time() - t0, 1), cls=type(model).__name__, params=model.num_parameters())

    from llmcompressor.modeling.moe.linearize import get_non_linearized_moes, linearize_moe

    fused = len(get_non_linearized_moes(model))
    if fused:
        t0 = time.time()
        linearize_moe(model)  # before derive(): the ignore list must see the per-expert Linear modules
        emit("linearize", fused_expert_modules=fused, seconds=round(time.time() - t0, 1))
    r = derive(model, cfg, [p for p in a.ignore.split(",") if p])
    emit("recipe", **r)
    if not r["sequential_target"]:
        emit("refused", reason="no decoder layer list found (no ModuleList outside vision modules)", hint="")
        return EXIT_REFUSED

    emit("step", n=2)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    ds, stats = calibration(a, tok)
    emit("calib", **stats)

    emit("step", n=3)
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.transform import SmoothQuantModifier

    preset = PRESETS[a.checkpoint]
    gptq = GPTQModifier(targets="Linear", scheme=preset, ignore=r["ignore"])
    recipe = [SmoothQuantModifier(smoothing_strength=0.8), gptq] if preset == "W8A8" else gptq
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    oneshot(model=model, processor=tok, dataset=ds, recipe=recipe, max_seq_length=a.seq_len,
            num_calibration_samples=len(ds), sequential_targets=[r["sequential_target"]])
    oneshot_seconds = round(time.time() - t0, 1)
    emit("oneshot", seconds=oneshot_seconds, peak_gpu_bytes=torch.cuda.max_memory_allocated() if cuda else None)

    cast = None
    orig = orig_dtype(cfg)
    if orig in ("bfloat16", "float16") and orig != a.compute_dtype:
        model.to(getattr(torch, orig))  # float32 compute: store unquantized tensors in the source dtype (spike Q6)
        cast = f"{a.compute_dtype}->{orig}"
    out = Path(a.out)
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    copied = copy_side_files(a.model, out)
    if r["mtp"]:
        from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
        save_mtp_tensors_to_checkpoint(source_model=a.model, dest_dir=str(out))
    import compressed_tensors
    import llmcompressor

    (out / "qtool.json").write_text(json.dumps({
        "signature": json.loads(a.signature),
        "recipe": {k: r[k] for k in ("ignore", "sequential_target", "num_layers", "linears_quantized")},
        "calib": {k: v for k, v in stats.items() if k != "cached"},
        "compute_dtype": a.compute_dtype, "cast": cast, "oneshot_seconds": oneshot_seconds,
        "versions": {"llmcompressor": llmcompressor.__version__, "transformers": transformers.__version__,
                     "torch": torch.__version__, "compressed_tensors": compressed_tensors.__version__},
    }, indent=1))
    emit("saved", path=str(out), bytes=dir_bytes(out), cast=cast, mtp=r["mtp"], copied=copied)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to see it pass**

Run: `./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh tests/container/test_quantize.py'`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/container/test_quantize.py images/scripts/quantize.py
git commit -m "Add container quantize script with events, calib cache and FP32 cast-back"
```

---

### Task 11: images/scripts/probe.py — metadata model cho preflight

Chạy được trong cả 2 image. Đọc config (không cần remote code), tokenizer (chế độ eval chat / completion), số tham số từ header safetensors (local) hoặc metadata Hub — **không tải weight**; tuỳ chọn kiểm task lm_eval (`TaskManager().all_tasks`, chỉ có trong image eval, ~30 s).

**Files:**
- Create: `images/scripts/probe.py`
- Test: `tests/container/test_probe.py`

**Interfaces:**
- Consumes: `recipe.eval_chat_mode/is_multimodal/orig_dtype/remote_code_required`.
- Produces: một dòng `@qtool {event: probe, model, architectures, model_type, dtype, multimodal, remote_code, quantized, quantization_config, params, weight_bytes, chat_mode, chat_reason, tasks_missing}` hoặc `{event: probe_error, message}` + exit 3.

Test chạy **trong image compress trên rtx4070 (CPU)**: `./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh <file>'`. Lần đầu script build image `qtool-compress:cu129-<hash>` (chỉ thêm layer pytest nếu cache spike còn). Sau task này chạy cả thư mục: `scripts/test-container.sh` → `17 passed`.

- [ ] **Step 1: Write the failing test** — `tests/container/test_probe.py`

```python
# ABOUTME: Container-tier tests for probe.py on local tiny model dirs: metadata, parameter count from
# ABOUTME: safetensors headers, chat-mode decision, remote-code and already-quantized detection.
import json

import pytest

from tinyhelpers import PARTS_ONLY_TEMPLATE, events

pytestmark = pytest.mark.container


def probe(capsys, model):
    import probe as p

    code = p.main(["--model", str(model)])
    return code, events(capsys.readouterr().out)[-1]


def test_probe_phi3(capsys, tiny_dir):
    code, e = probe(capsys, tiny_dir("phi3"))
    assert code == 0 and e["event"] == "probe"
    assert e["architectures"] == ["Phi3ForCausalLM"] and e["dtype"] == "bfloat16"
    assert e["quantized"] is False and e["remote_code"] is False and e["multimodal"] is False
    assert e["params"] > 0 and e["weight_bytes"] >= e["params"] * 2
    assert (e["chat_mode"], e["tasks_missing"]) == ("chat", [])


def test_probe_template_dropping_content_means_completion(capsys, tiny_dir):
    code, e = probe(capsys, tiny_dir("phi3", PARTS_ONLY_TEMPLATE))
    assert e["chat_mode"] == "completion"


def test_probe_multimodal(capsys, tiny_dir):
    code, e = probe(capsys, tiny_dir("qwen3_5"))
    assert e["multimodal"] is True and e["architectures"] == ["Qwen3_5ForConditionalGeneration"]


def test_probe_remote_code(capsys, tiny_dir, tmp_path):
    src = tiny_dir("phi3")
    d = tmp_path / "remote"
    d.mkdir()
    cfg = json.loads((src / "config.json").read_text())
    cfg.update(architectures=["MadeUpForCausalLM"], auto_map={"AutoModelForCausalLM": "modeling_x.MadeUpForCausalLM"})
    (d / "config.json").write_text(json.dumps(cfg))
    code, e = probe(capsys, d)
    assert e["remote_code"] is True and e["chat_mode"] == "completion"


def test_probe_error_is_reported(capsys, tmp_path):
    code, e = probe(capsys, tmp_path / "missing")
    assert code == 3 and e["event"] == "probe_error"
```

- [ ] **Step 2: Run the test to see it fail**

Run: `./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh tests/container/test_probe.py'`
Expected: FAIL — `ModuleNotFoundError: No module named 'probe'`

- [ ] **Step 3: Write the implementation** — `images/scripts/probe.py`

```python
# ABOUTME: Preflight probe run inside a qtool container: reads a model's config, tokenizer and safetensors headers
# ABOUTME: (no weights downloaded) and optionally checks lm_eval task names; prints one `@qtool {"event": "probe"}` line.
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipe import eval_chat_mode, is_multimodal, orig_dtype, remote_code_required  # noqa: E402

EXIT_PROBE_ERROR = 3


def emit(event: str, **fields) -> None:
    print("@qtool " + json.dumps({"event": event, **fields}, default=str), flush=True)


def header_params(path: Path) -> int:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")


def weights_info(model: str) -> tuple[int | None, int | None]:
    """(parameter count, bytes of *.safetensors) from local headers or Hub metadata; never downloads weights."""
    if os.path.isdir(model):
        files = sorted(Path(model).glob("*.safetensors"))
        if not files:
            return None, None
        return sum(header_params(f) for f in files), sum(f.stat().st_size for f in files)
    from huggingface_hub import HfApi, get_safetensors_metadata

    params = sum(get_safetensors_metadata(model).parameter_count.values())
    info = HfApi().model_info(model, files_metadata=True)
    size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    return params, size or None


def missing_tasks(names: list[str]) -> list[str]:
    from lm_eval.tasks import TaskManager

    known = set(TaskManager().all_tasks)
    return [t for t in names if t not in known]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="", help="comma-separated lm_eval tasks to check (eval image only)")
    a = ap.parse_args(argv)
    import transformers

    try:
        cfg, _ = transformers.PretrainedConfig.get_config_dict(a.model)
        remote = remote_code_required(cfg, transformers)
        mode, reason = ("completion", "remote-code tokenizer not loaded") if remote else \
            eval_chat_mode(transformers.AutoTokenizer.from_pretrained(a.model))
        params, size = weights_info(a.model)
    except Exception as e:  # noqa: BLE001 - report the original error text to the host verbatim
        emit("probe_error", message=f"{type(e).__name__}: {e}")
        return EXIT_PROBE_ERROR
    qc = cfg.get("quantization_config")
    emit("probe", model=a.model, architectures=cfg.get("architectures"), model_type=cfg.get("model_type"),
         dtype=orig_dtype(cfg), multimodal=is_multimodal(cfg), remote_code=remote, quantized=qc is not None,
         quantization_config=qc, params=params, weight_bytes=size, chat_mode=mode, chat_reason=reason,
         tasks_missing=missing_tasks([t for t in a.tasks.split(",") if t]) if a.tasks else [])
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to see it pass**

Run: `./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh tests/container/test_probe.py'`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/container/test_probe.py images/scripts/probe.py
git commit -m "Add container probe script for preflight metadata"
```

---

### Task 12: runner.py — chạy container, stream log, Ctrl-C

Docker giả trong test = tiến trình Python phát lại log thật. Ctrl-C: `docker stop` rồi **tiếp tục đọc pipe tới EOF** (không thì `docker run` kẹt khi pipe đầy — lỗi treo bắt được khi viết plan), quá 60 s thì kill.

**Files:**
- Create: `qtool/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `ContainerSpec(image, command, name, gpu=None, env={}, mounts=[], memory_bytes=None)` (env value `None` = truyền biến host), `docker_argv(spec)`, `LineSplitter.feed/flush`, `run_container(spec, log_path, on_line, on_tick=None, popen, tick=1.0, stop) -> int`, `atomic_dir(final)` (context manager, `.tmp-<name>`).

- [ ] **Step 1: Write the failing test** — `tests/test_runner.py`

```python
# ABOUTME: Tests for qtool.runner with a fake docker (a Python child replaying a real fixture log):
# ABOUTME: argv building, line splitting with "\r", raw log, exit codes, Ctrl-C stopping the container, atomic dirs.
import subprocess
import sys

import pytest

from qtool import runner
from qtool.parsers import QuantizeProgress


def spec(**over):
    return runner.ContainerSpec(**{"image": "img:1", "command": ["python", "/qtool/quantize.py"], "name": "qtool-1-0", **over})


def test_docker_argv_full():
    s = spec(gpu=1, memory_bytes=25_000_000_000, env={"HF_HOME": "/hf", "HF_TOKEN": None},
             mounts=[("/home/u/.cache/huggingface", "/hf", False), ("/repo/images/scripts", "/qtool", True)])
    assert runner.docker_argv(s) == [
        "docker", "run", "--rm", "--name", "qtool-1-0", "--gpus", "device=1",
        "--memory", "25000000000", "--memory-swap", "25000000000", "-e", "HF_HOME=/hf", "-e", "HF_TOKEN",
        "-v", "/home/u/.cache/huggingface:/hf", "-v", "/repo/images/scripts:/qtool:ro",
        "img:1", "python", "/qtool/quantize.py"]


def test_docker_argv_minimal():
    assert runner.docker_argv(spec()) == ["docker", "run", "--rm", "--name", "qtool-1-0", "img:1", "python",
                                          "/qtool/quantize.py"]


def test_line_splitter_across_chunks():
    s = runner.LineSplitter()
    assert s.feed("(1/33): Calib") == []
    assert s.feed("rating: 1/512\r(1/33): Calibrating: 2/512\r") == ["(1/33): Calibrating: 1/512", "(1/33): Calibrating: 2/512"]
    assert s.feed("\nlast") == []
    assert s.flush() == ["last"]


def fake_popen(log_file, code=0):
    script = (f"import sys; sys.stdout.buffer.write(open({str(log_file)!r}, 'rb').read()); "
              f"sys.stdout.flush(); sys.exit({code})")

    def popen(argv, **kw):
        assert argv[:2] == ["docker", "run"]
        return subprocess.Popen([sys.executable, "-c", script], **kw)

    return popen


def test_run_container_replays_real_log(fixtures, tmp_path):
    src = fixtures / "llmcompressor/w4a16-gptq.log"
    progress, ticks = QuantizeProgress(), []
    log = tmp_path / "logs" / "INT4-W4A16.quantize.log"
    code = runner.run_container(spec(), log, progress.feed, on_tick=lambda: ticks.append(1), popen=fake_popen(src))
    assert code == 0
    assert log.read_bytes() == src.read_bytes()
    assert (progress.subgraph, progress.total, progress.gptq_modules) == (33, 33, 128)
    assert ticks


def test_run_container_returns_exit_code(fixtures, tmp_path):
    code = runner.run_container(spec(), tmp_path / "x.log", lambda line: None,
                                popen=fake_popen(fixtures / "llmcompressor/w4a16-fp32-oom-killed.log", 137))
    assert code == 137


def test_ctrl_c_stops_container_and_reraises(fixtures, tmp_path):
    stopped = []

    def on_line(line):
        if "(2/33)" in line:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner.run_container(spec(name="qtool-7-3"), tmp_path / "x.log", on_line,
                             popen=fake_popen(fixtures / "llmcompressor/w4a16-gptq.log"), stop=stopped.append)
    assert stopped == ["qtool-7-3"]


def test_atomic_dir_renames_on_success_and_cleans_on_failure(tmp_path):
    final = tmp_path / "checkpoints" / "INT4-W4A16"
    with runner.atomic_dir(final) as tmp:
        assert tmp == tmp_path / "checkpoints" / ".tmp-INT4-W4A16"
        (tmp / "config.json").write_text("{}")
    assert (final / "config.json").exists() and not tmp.exists()
    with pytest.raises(RuntimeError):
        with runner.atomic_dir(final) as tmp:
            (tmp / "partial").write_text("x")
            raise RuntimeError("container failed")
    assert not tmp.exists() and (final / "config.json").exists()
```

- [ ] **Step 2: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_runner.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool.runner'`

- [ ] **Step 3: Write the implementation** — `qtool/runner.py`

```python
# ABOUTME: Runs containers: builds the `docker run` argv, streams merged stdout/stderr line by line (tqdm "\r" too),
# ABOUTME: appends the raw bytes to a log, calls a tick callback, stops the container on Ctrl-C; plus atomic dirs.
import codecs
import os
import re
import selectors
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContainerSpec:
    image: str
    command: list[str]
    name: str
    gpu: int | None = None
    env: dict = field(default_factory=dict)  # value None: pass the host variable through (-e NAME)
    mounts: list = field(default_factory=list)  # (host path, container path, read_only)
    memory_bytes: int | None = None


def docker_argv(spec: ContainerSpec) -> list[str]:
    argv = ["docker", "run", "--rm", "--name", spec.name]
    if spec.gpu is not None:
        argv += ["--gpus", f"device={spec.gpu}"]
    if spec.memory_bytes:
        argv += ["--memory", str(spec.memory_bytes), "--memory-swap", str(spec.memory_bytes)]
    for k, v in spec.env.items():
        argv += ["-e", k if v is None else f"{k}={v}"]
    for host, ctr, ro in spec.mounts:
        argv += ["-v", f"{host}:{ctr}" + (":ro" if ro else "")]
    return [*argv, spec.image, *spec.command]


class LineSplitter:
    def __init__(self):
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        parts = re.split(r"\r\n|\r|\n", self._buf + text)
        self._buf = parts.pop()
        return [p for p in parts if p]

    def flush(self) -> list[str]:
        rest, self._buf = self._buf, ""
        return [rest] if rest else []


def _docker_stop(name: str) -> None:
    subprocess.run(["docker", "stop", "-t", "10", name], capture_output=True)


def run_container(spec: ContainerSpec, log_path: Path, on_line, on_tick=None, popen=subprocess.Popen,
                  tick: float = 1.0, stop=_docker_stop) -> int:
    """Run to completion and return the exit code; on KeyboardInterrupt stop the container and re-raise."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    splitter = LineSplitter()
    with open(log_path, "ab") as log:
        proc = popen(docker_argv(spec), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                if sel.select(timeout=tick):
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    for line in splitter.feed(decoder.decode(chunk)):
                        on_line(line)
                if on_tick:
                    on_tick()
            for line in splitter.feed(decoder.decode(b"", final=True)) + splitter.flush():
                on_line(line)
            return proc.wait()
        except KeyboardInterrupt:
            stop(spec.name)
            _drain(proc, sel, log)
            raise
        finally:
            sel.close()
            proc.stdout.close()


def _drain(proc, sel, log, deadline_s: float = 60.0) -> None:
    """Keep reading after `docker stop` so `docker run` is never blocked on a full pipe; kill it past the deadline."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if sel.select(timeout=1.0):
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            log.write(chunk)
    else:
        proc.kill()
    proc.wait()


@contextmanager
def atomic_dir(final: Path):
    """Yield <parent>/.tmp-<name>; rename to final only if the block succeeds, else delete it."""
    final = Path(final)
    tmp = final.with_name(".tmp-" + final.name)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
```

- [ ] **Step 4: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_runner.py`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_runner.py qtool/runner.py
git commit -m "Add container runner with line streaming, Ctrl-C handling and atomic dirs"
```

---

### Task 13: flow.py — điều phối quantize / eval

Session giữ GPU, thư mục, image; mọi container đi qua `Session.run_container` (chạy root, rồi `chown` trả file cho dev). Render sự kiện thành log console §8, ETA `est.` từ `(N/M)`, `--on-error`, lỗi step 1–2 dừng toàn bộ, scheme phụ thuộc checkpoint hỏng → SKIPPED, baseline cache theo task, baseline không vừa VRAM → SKIPPED + `n/a`, `--act` không có Marlin → INVALID, bảng tổng hợp + `summary.json`.

**Files:**
- Create: `tests/fakes.py`
- Create: `qtool/flow.py`
- Test: `tests/test_flow.py`

**Interfaces:**
- Consumes: tất cả module trên.
- Produces: `VERIFIED_ARCHS`, `StepFailed(reason, step)`, `Session(console, gpu, rd, hf_cache, home, user, compress_image, eval_image, memory_bytes, run, chown, clock)` + `.header()`, `.name()`, `.run_container()`, `.current`, `.started`; `mount_path`, `container_model`, `container_spec`, `failure_reason`, `dir_bytes`, `probe(s, image, model, tasks) -> dict`, `QuantizeOptions`, `quantize_checkpoint`, `EvalJob`, `EvalResult`, `model_args`, `quantization_text`, `save_task_results`, `eval_job`, `baseline_row`, `print_summary`, `QuantizeRun`, `run_quantize(s, r) -> int`, `EvalRun`, `run_eval(s, r) -> int`, `label_for(model, act, quantization)`.
- Test double: `tests/fakes.py` → `GPU`, `W4A16_QC`, `W8A8_QC`, `ev()`, `arg()`, `FakeDocker` (`quantize`, `evals`, `probes`, `specs`).

- [ ] **Step 1: Write the shared fake docker** — `tests/fakes.py`

```python
# ABOUTME: Test doubles shared by flow and CLI tests: a fake `docker run` replaying real spike logs and writing the
# ABOUTME: files real containers would (checkpoints, lm_eval results, probe event), plus common constants.
import json
import shutil

from qtool import parsers, preflight

GPU = preflight.Gpu(0, "NVIDIA GeForce RTX 4070 Ti SUPER", "8.9", 16, 16376, 15928)
W4A16_QC = {"quant_method": "compressed-tensors", "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int", "symmetric": True,
                                                      "strategy": "group", "group_size": 128}}}}
W8A8_QC = {"quant_method": "compressed-tensors", "format": "int-quantized",
           "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel"},
                                         "input_activations": {"num_bits": 8, "type": "int", "dynamic": True}}}}


def ev(event, **fields):
    return "@qtool " + json.dumps({"event": event, **fields})


def arg(cmd, name):
    return cmd[cmd.index(name) + 1]


class FakeDocker:
    """Plays the role of `docker run`: replays fixture logs, writes outputs under the mounted runs dir."""

    def __init__(self, fixtures, rd):
        self.fixtures, self.rd = fixtures, rd
        self.specs = []
        self.quantize = {}  # checkpoint -> "ok" | "fail-step2" | "fail-step3" | "killed"
        self.probes = {}  # model -> probe event fields (defaults: Qwen3.5-4B BF16)
        self.evals = {}  # label -> (log fixture, results dir fixture or None, exit code)

    def host(self, ctr_path):
        return self.rd / ctr_path.removeprefix("/runs/")

    def lines(self, rel):
        return parsers.split_lines((self.fixtures / rel).read_text(errors="replace"))

    def __call__(self, spec, log_path, on_line, on_tick=None):
        self.specs.append(spec)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = spec.command
        if "/qtool/probe.py" in cmd:
            model = arg(cmd, "--model")
            host = {ctr: h for h, ctr, _ in spec.mounts}.get(model, model)  # local dirs are mounted at /model
            out, code = [ev("probe", **self.probe_fields(host))], 0
        elif "/qtool/quantize.py" in cmd:
            out, code = self.run_quantize(cmd)
        else:
            out, code = self.run_eval(cmd)
        for line in out:
            on_line(line)
            if on_tick:
                on_tick()
        return code

    def probe_fields(self, model):
        base = {"model": model, "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
                "dtype": "bfloat16", "multimodal": True, "remote_code": False, "quantized": False,
                "quantization_config": None, "params": 4_539_265_536, "weight_bytes": 9_078_531_072,
                "chat_mode": "chat", "chat_reason": "chat template keeps message content", "tasks_missing": []}
        return {**base, **self.probes.get(model, {})}

    def run_quantize(self, cmd):
        ckpt, mode = arg(cmd, "--checkpoint"), self.quantize.get(arg(cmd, "--checkpoint"), "ok")
        out = [ev("step", n=1),
               ev("config", architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5", dtype="bfloat16",
                  multimodal=True, loader="AutoModelForImageTextToText"),
               ev("load", seconds=4.7, cls="Qwen3_5ForConditionalGeneration", params=4539265536),
               ev("recipe", ignore=["re:.*lm_head", "re:.*visual.*", "re:.*linear_attn.*"],
                  why={"re:.*lm_head": "output head", "re:.*visual.*": "vision/projector",
                       "re:.*linear_attn.*": "hybrid linear-attention / SSM mixer"},
                  sequential_target="Qwen3_5DecoderLayer", num_layers=32, num_experts=None, mtp=True, linears_total=347,
                  linears_quantized=128, quantized_sample=["model.language_model.layers.N.mlp.down_proj",
                                                           "model.language_model.layers.N.self_attn.q_proj"]),
               ev("step", n=2)]
        if mode == "fail-step2":
            return out + ["Traceback (most recent call last):", "datasets.exceptions.DatasetNotFoundError: no such dataset"], 1
        out += [ev("calib", source="HuggingFaceH4/ultrachat_200k:train_sft", template="chat_template(str)", n=512, min=199,
                   median=1080, max=2048, truncated=48, cached=False, seconds=8.2), ev("step", n=3)]
        if mode == "fail-step3":
            return out + self.lines("llmcompressor/w4a16-error-no-cc.log"), 1
        if mode == "killed":
            return out + self.lines("llmcompressor/w4a16-fp32-oom-killed.log")[-50:], 137
        log = "llmcompressor/w8a8-smoothquant-gptq.log" if ckpt == "INT8-W8A8" else "llmcompressor/w4a16-gptq.log"
        out += [line for line in self.lines(log) if "(" in line or "|" in line]
        tmp = self.host(arg(cmd, "--out"))
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "config.json").write_text(json.dumps({"quantization_config": W8A8_QC if ckpt == "INT8-W8A8" else W4A16_QC}))
        (tmp / "model.safetensors").write_bytes(b"x" * 1000)
        (tmp / "qtool.json").write_text(json.dumps({"signature": json.loads(arg(cmd, "--signature"))}))
        out += [ev("oneshot", seconds=905.9, peak_gpu_bytes=2097972224),
                ev("saved", path=str(tmp), bytes=5_200_000_000, cast=None, mtp=True, copied=[])]
        return out, 0

    def run_eval(self, cmd):
        label = arg(cmd, "--output_path").rsplit(".tmp-", 1)[1]
        log, results, code = self.evals[label]
        if results:
            dest = self.host(arg(cmd, "--output_path")) / "model"
            dest.mkdir(parents=True)
            src = parsers.find_results_json(self.fixtures / results)
            shutil.copy(src, dest / src.name)
        return self.lines(log), code
```

- [ ] **Step 2: Write the failing test** — `tests/test_flow.py`

```python
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
                    "int8-w8a8": ("lm_eval/eval-w8a8.log", "lm_eval/out/w8a8", 0),
                    "fp8-w8a8": ("lm_eval/eval-fp8-online.log", "lm_eval/out/fp8-online", 0)}
    return s, docker, out


def quantize_run(s, names, no_eval=False, on_error="continue"):
    needed = flow.schemes.checkpoints_for(names)
    sigs = {c: plan.checkpoint_signature("Qwen/Qwen3.5-4B", c, "HuggingFaceH4/ultrachat_200k:train_sft", 512, 2048, [],
                                         "0.13.0") for c in needed}
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
    assert "  cache        : gsm8k: cache: valid, reused" in out.getvalue()


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
```

- [ ] **Step 3: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_flow.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtool.flow'`

- [ ] **Step 4: Write the implementation** — `qtool/flow.py`

```python
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
    c.kv("recipe", f"{schemes.CHECKPOINTS[ckpt]['recipe']} · sequential {r['sequential_target']}")
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
        if st["step"] != 3 or not progress.total or st["oneshot_done"]:
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
    c.done(s.clock() - st["t"])
    return st["saved"] or {"bytes": dir_bytes(s.rd / "checkpoints" / ckpt)}


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
    raw = json.loads(path.read_text())
    c.kv("lm_eval", f"{p.prompts_done}/{p.prompts_total} prompts")
    for line in p.table:
        c.line("  " + line)
    metrics = parsers.metrics_from(raw["results"])
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
    for name in r.schemes:
        sc = schemes.SCHEMES[name]
        if sc.checkpoint and not ok.get(sc.checkpoint):
            rows.append(report.Row(name, "SKIPPED", note=f"{sc.checkpoint} failed"))
            continue
        if sc.checkpoint:
            path = s.rd / "checkpoints" / sc.checkpoint
            shared = sum(1 for x in r.schemes if schemes.SCHEMES[x].checkpoint == sc.checkpoint) > 1
            qc = json.loads((path / "config.json").read_text()).get("quantization_config")
            model, weight = str(path), sizes[sc.checkpoint]
            info = [("checkpoint", f"checkpoints/{sc.checkpoint}" + (" (shared)" if shared else "")),
                    ("quantization", quantization_text(qc))]
        else:
            shared, model, weight = False, r.model, r.params if r.params else None  # FP8 online: ~1 byte per param
            info = [("model", f"{r.model} (BF16)"), ("quantization", f"vLLM online {sc.quantization}")]
        if sc.env:
            info.append(("runtime env", " ".join(f"{k}={v}" for k, v in sc.env.items())))
        skip = preflight.check_eval_fit(name, weight, s.gpu)
        if skip:
            s.header(f"[4/4] Evaluating {name}")
            c.warn(skip)
            rows.append(report.Row(name, "SKIPPED", size_bytes=weight, note="does not fit"))
            continue
        metas = lambda t, name=name: plan.eval_meta(name, t, r.limit, s.eval_image, r.mode, s.gpu.name)  # noqa: E731
        job = EvalJob(name, model, name, sc.env, sc.quantization, info, expect_marlin=bool(sc.env))
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
```

- [ ] **Step 5: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_flow.py`
Expected: `15 passed`

- [ ] **Step 6: Commit**

```bash
git add Write the shared fake docker tests/test_flow.py qtool/flow.py
git commit -m "Add run orchestration: quantize and eval jobs, caches, on-error, summary"
```

---

### Task 14: cli.py đầy đủ — 2 lệnh con, preflight, exit code

Thay `qtool/cli.py` và `tests/test_cli.py` tối thiểu của Task 1. `System` gom mọi tác động lên host (docker, nvidia-smi, /proc, build image, chown) để test thay bằng `FakeSystem`. Thứ tự: validate → GPU → `--dry-run` (in kế hoạch, không probe) → lock → build image nếu thiếu → probe → findings (✗ = exit 2) → kế hoạch → checkpoint stale = exit 2 → chạy.

**Files:**
- Modify (replace whole file): `qtool/cli.py`
- Modify (replace whole file): `tests/test_cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `flow.*`, `preflight.*`, `plan.*`, `images.*`, `schemes.*`.
- Produces: `main(argv=None, system=None, console=None) -> int`, `build_parser()`, `System`, `cmd_quantize`, `cmd_eval`, `probe_findings(info, quantize, no_eval=False)`, `UsageError`, `DEFAULT_CALIB`.

- [ ] **Step 1: Write the failing test** — `tests/test_cli.py`

```python
# ABOUTME: Tests for qtool.cli: argument validation (exit 2), dry-run, preflight blocking before any container,
# ABOUTME: a full fake quantize run, stale checkpoint, Ctrl-C (exit 130) and `qtool eval --act` checks.
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
        ticks = itertools.count(0, 5)
        self.clock = lambda: next(ticks)

    def run(self, spec, log_path, on_line, on_tick=None):
        if self.docker is None:
            self.docker = FakeDocker(self.fixtures, Path(spec.mounts[3][0]))
            self.docker.evals = {"bf16": ("lm_eval/eval-bf16-chat.log", "lm_eval/out/bf16-chat", 0),
                                 "int4-w4a16": ("lm_eval/eval-w4a16-none.log", "lm_eval/out/w4a16-none", 0),
                                 "INT4-W4A16+act-int8": ("lm_eval/eval-w4a16-int8.log", "lm_eval/out/w4a16-int8", 0)}
            self.docker.probes = getattr(self, "probes", {})
        return self.docker(spec, log_path, on_line, on_tick)

    def chown(self, argv, **kw):
        self.chowns.append(argv)

    def docker_error(self):
        return None

    def gpus(self):
        f = self.fixtures / "nvidia-smi"
        apps = f / ("apollo-8gpu-apps.csv" if self.gpu_csv.startswith("apollo") else "rtx4070-apps.csv")
        uuid = f / ("apollo-8gpu-uuid.csv" if self.gpu_csv.startswith("apollo") else "rtx4070-uuid.csv")
        return preflight.parse_gpus((f / f"{self.gpu_csv}.csv").read_text()), \
            preflight.parse_apps(apps.read_text(), uuid.read_text())

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
```

- [ ] **Step 2: Run the test to see it fail**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_cli.py`
Expected: FAIL — `AttributeError: module 'qtool.cli' has no attribute 'System'` (và các test mới khác)

- [ ] **Step 3: Write the implementation** — `qtool/cli.py`

```python
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
        return preflight.query_gpus()

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
    gpu = preflight.select_gpu(gpus, args.gpu, apps)
    console.line(f"qtool {__version__} · {time.strftime('%Y-%m-%d %H:%M:%S')} · {gpu.name} "
                 f"{gpu.total_mib / 1024:.0f} GB SM {gpu.cc}")
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
    if args.dry_run:
        for line in plan.render_plan(args.model, states, names, tasks, args.eval_limit, "cache not checked in --dry-run",
                                     args.no_eval):
            console.line(line)
        return 0
    lock = preflight.RunLock(system.lock_path)
    lock.acquire()
    s = None
    try:
        for d in (rd / "logs", output / ".home"):
            d.mkdir(parents=True, exist_ok=True)
        compress = system.ensure_image("compress", images.compress_cuda(gpu.cc), console)
        evali = None if args.no_eval else system.ensure_image("eval", None, console)
        s = flow.Session(console, gpu, rd, Path(args.hf_cache).expanduser(), output / ".home", system.user(), compress,
                         evali, run=system.run, chown=system.chown, clock=system.clock)
        s.started = s.clock()
        info = flow.probe(s, evali or compress, args.model, [] if args.no_eval else tasks)
        ram = preflight.mem_available_bytes(system.meminfo())
        findings = probe_findings(info, True, args.no_eval) + preflight.check_quantize(
            gpu, info["params"], dtype, ram, names, args.no_eval, thresholds=not args.skip_preflight)
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
        if stale := [st for st in states if st.action == "stale"]:
            for st in stale:
                console.line(f"error: checkpoints/{st.name} was made with different settings: " + "; ".join(st.diff))
            console.line("re-run with --force to rebuild it")
            return 2
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
    if args.dry_run:
        console.line(f"Plan for eval of {args.model}")
        if args.baseline:
            console.line(f"  eval      baseline {args.baseline} (cache not checked in --dry-run)")
        console.line(f"  eval      {label}   tasks: {','.join(tasks)} ({args.eval_limit or 'full'})")
        return 0
    lock = preflight.RunLock(system.lock_path)
    lock.acquire()
    s = None
    try:
        for d in (rd / "logs", output / ".home"):
            d.mkdir(parents=True, exist_ok=True)
        evali = system.ensure_image("eval", None, console)
        s = flow.Session(console, gpu, rd, Path(args.hf_cache).expanduser(), output / ".home", system.user(), None, evali,
                         run=system.run, chown=system.chown, clock=system.clock)
        s.started = s.clock()
        info = flow.probe(s, evali, args.model, tasks)
        binfo = flow.probe(s, evali, args.baseline) if args.baseline else None
        rows = ["int4-w4a8-" + args.act] if args.act else (["fp8-w8a8"] if args.quantization else [])
        findings = probe_findings(info, False) + preflight.check_eval_sm(gpu, rows, baseline=bool(args.baseline))
        if args.act:
            findings += preflight.check_act(gpu, info["quantization_config"])
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
        bbytes = binfo["params"] * 2 if binfo and binfo["params"] else None
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
    except (UsageError, preflight.PreflightError) as e:
        console.line(f"error: {e}")
        return 2
```

- [ ] **Step 4: Run the test to see it pass**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_cli.py`
Expected: `18 passed`; toàn bộ: `uv run --no-project --with pytest python -m pytest -q` → `130 passed, 17 deselected`

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli.py qtool/cli.py
git commit -m "Add full CLI: quantize and eval subcommands with preflight wiring"
```

---

### Task 15: README, chạy toàn bộ test trên rtx4070

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README** — `README.md`

````markdown
# qtool

Quantize a Hugging Face model with LLM Compressor and evaluate it in vLLM — the 4 steps of the
[vLLM LLM Compressor guide](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/)
(load, calibrate, quantize, evaluate) as one command with step-by-step console output.

Design: `docs/specs/2026-10-30-qtool-design.md` · spike results: `docs/spike/2026-10-30-findings.md`.

## Requirements

- NVIDIA GPU, Docker with NVIDIA Container Toolkit, Python ≥ 3.10 on the host (standard library only).
- Images are built on first use: `qtool-compress` (torch 2.13, llmcompressor 0.13.0, transformers 5.14.1;
  `cu126` variant for SM < 7.5) and `qtool-eval` (vLLM 0.28 + lm-eval 0.4.13).

## Install

```bash
ln -s "$PWD/bin/qtool" ~/.local/bin/qtool
```

## Use

```bash
# quantize into two schemes sharing one INT4 checkpoint, then evaluate both against BF16 on gsm8k
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16,int4-w4a8-fp8

# only evaluate (any model vLLM can load), with a cached baseline for Δ pp / recovery %
qtool eval RedHatAI/Qwen3.5-4B-quantized.w4a16 --act fp8 --baseline Qwen/Qwen3.5-4B

# V100 (SM 7.0): quantize only, evaluate the checkpoint on another GPU later
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16 --no-eval --gpu 0
```

Schemes: `int4-w4a16`, `int4-w4a8-fp8`, `int4-w4a8-int8` (INT4 checkpoint + Marlin W4A8 at runtime, SM 8.9 / 12.0),
`int8-w8a8` (SmoothQuant + GPTQ), `fp8-w8a8` (vLLM online FP8, no checkpoint).

Results live in `~/qtool/runs/<model>/` (`checkpoints/`, `results/`, `logs/`). qtool never stops or starts other
containers or services: preflight only reports what is using the GPU.

## Develop

```bash
uv run --no-project --with pytest python -m pytest      # host tests (no GPU, no Docker)
./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh'   # container tests (CPU, compress image)
```

GPU integration and acceptance runs are manual (see `docs/plans/2026-11-01-qtool-implementation.md`, tasks 16–17).
````

- [ ] **Step 2: Toàn bộ test host (local + rtx4070) và container**

```bash
uv run --no-project --with pytest python -m pytest -q                                   # local: 130 passed, 17 deselected
./sync.sh push
ssh rtx4070 'cd ~/qtool && uv run --no-project --with pytest python -m pytest -q && scripts/test-container.sh'   # 130 passed + 17 passed
```

Expected: không cảnh báo, không lỗi ở cả 3 lệnh.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add README"
```

---

### Task 16: Integration trên GPU (chạy tay, rtx4070)

Mục đích: chạy thật đầu-cuối. Trước mỗi bước: `ssh rtx4070 nvidia-smi` — GPU chỉ có `trt-export-server` (~210 MiB); bận thứ khác → dừng, báo Nhân (tool không dừng service).

- [ ] **Step 1: Cài shim + smoke model tí hon (không cần mạng)**

```bash
ssh rtx4070 'ln -sf ~/qtool/bin/qtool ~/.local/bin/qtool'
ssh rtx4070 'mkdir -p ~/qtool-smoke && cd ~/qtool && docker run --rm -i --user $(id -u):$(id -g) -e HOME=/tmp -v ~/qtool:/src:ro -v ~/qtool-smoke:/out -w /src/tests/container $(python3 -c "from qtool import images; print(images.tag(\"compress\", \"cu129\"))") python -' <<'EOF'
import json, sys
sys.path.insert(0, ".")
from tinyhelpers import CORPUS, build_model, make_tokenizer
model, _ = build_model("phi3")
model.save_pretrained("/out/tiny-phi3")
make_tokenizer().save_pretrained("/out/tiny-phi3")
with open("/out/calib.jsonl", "w") as f:
    for t in CORPUS:
        f.write(json.dumps({"messages": [{"role": "user", "content": t}, {"role": "assistant", "content": "ok"}]}) + "\n")
EOF
ssh rtx4070 'qtool quantize ~/qtool-smoke/tiny-phi3 --scheme int4-w4a16,int8-w8a8 --calib ~/qtool-smoke/calib.jsonl --calib-samples 16 --calib-seq-len 64 --no-eval --output-dir ~/qtool-smoke/runs; echo EXIT $?; find ~/qtool-smoke/runs ! -user $(id -un) | wc -l'
```

Expected: `EXIT 0`; đủ header `[1/4]`, `[2/4]`, `[3/4]` cho mỗi checkpoint; checkpoint thứ 2 in `cache        : reused from cache`; bảng `Summary · checkpoints` 2 dòng `OK`; số file không thuộc user = `0`.

- [ ] **Step 2: Qwen3.5-4B thật, 2 scheme dùng chung checkpoint, eval limit 20 (~25 phút)**

```bash
ssh rtx4070 'qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16,int4-w4a8-fp8 --eval-limit 20 2>&1 | tee ~/qtool-it.log; echo EXIT ${PIPESTATUS[0]}'
```

Expected (đối chiếu spike): `EXIT 0`; preflight `need 5.4 GB` VRAM / `14.5 GB` RAM; `[3/4]` có `ignore : lm_head (output head) · visual (vision/projector) · linear_attn (…)`, `quantized : 128 Linear`, ETA `est.`, `saved : checkpoints/INT4-W4A16 · 9.1 GB → ~5.2 GB · MTP saved`; eval in `kernel : MarlinLinearKernel for CompressedTensorsWNA16` cho cả 2 scheme, `runtime env : VLLM_MARLIN_INPUT_DTYPE=fp8` cho scheme thứ 2; bảng Summary 3 dòng OK.

- [ ] **Step 3: Chạy lại y hệt → cache**

Expected: `reuse INT4-W4A16`, baseline `cache: valid, reused`, không có container quantize; chỉ 2 lượt eval scheme.

- [ ] **Step 4: Ghi kết quả vào `docs/spike/2026-10-30-findings.md` (mục "Integration v0.1") và commit**

```bash
git add docs/spike/2026-10-30-findings.md
git commit -m "Record v0.1 integration run on rtx4070"
```

---

### Task 17: Acceptance (spec §9 tầng 4) rồi kết thúc nhánh

- [ ] **Step 1: Checkpoint RedHat + W4A8 fp8 (bake-off 26/10)**

```bash
ssh rtx4070 'qtool eval RedHatAI/Qwen3.5-4B-quantized.w4a16 --act fp8 --eval-limit 50; echo EXIT $?'
```

Expected: `EXIT 0`, `kernel : MarlinLinearKernel for CompressedTensorsWNA16`, `runtime env : VLLM_MARLIN_INPUT_DTYPE=fp8`, strict-match ≈ 0.78 (spike `rh-fp8`).

- [ ] **Step 2: Checkpoint tự làm vs RedHat trên gsm8k full (~25 phút)**

```bash
ssh rtx4070 'qtool eval ~/qtool/runs/Qwen3.5-4B/checkpoints/INT4-W4A16 --baseline RedHatAI/Qwen3.5-4B-quantized.w4a16; echo EXIT $?'
```

Expected: `EXIT 0`; |Δ| giữa 2 checkpoint ≤ stderr của lm_eval (≈ ±0.011 với 1319 câu). Lệch hơn → không sửa code, báo Nhân kèm 2 bảng lm_eval.

- [ ] **Step 3: Ghi kết quả acceptance vào findings, commit**

```bash
git add docs/spike/2026-10-30-findings.md
git commit -m "Record v0.1 acceptance results"
```

- [ ] **Step 4: Kết thúc nhánh** — REQUIRED SUB-SKILL: superpowers:finishing-a-development-branch (hỏi Nhân trước khi merge / dọn artifact spike).
