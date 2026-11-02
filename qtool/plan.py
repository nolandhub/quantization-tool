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
                         llmcompressor: str, compute_dtype: str) -> dict:
    return {"model": model, "checkpoint": ckpt, "recipe": schemes.CHECKPOINTS[ckpt]["recipe"], "ignore": sorted(ignore),
            "calib": calib, "calib_samples": samples, "calib_seq_len": seq_len, "llmcompressor": llmcompressor,
            "compute_dtype": compute_dtype}


def calib_key(model: str, calib: str, samples: int, seq_len: int) -> str:
    return hashlib.sha1(json.dumps([model, calib, samples, seq_len]).encode()).hexdigest()[:12]


def _show(v) -> str:
    return "full" if v is None else str(v)


def diff(saved: dict, wanted: dict, names: dict | None = None) -> list[str]:
    names = names or {}
    return [f"{names.get(k, k)}: {_show(saved[k]) if k in saved else 'not recorded'} → {_show(wanted[k])}"
            for k in wanted if saved.get(k) != wanted[k]]


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
            try:
                saved = json.loads(meta.read_text()).get("signature", {})
            except (OSError, json.JSONDecodeError) as e:
                out.append(CheckpointState(name, users, "stale", [f"qtool.json unreadable: {type(e).__name__}"]))
                continue
            d = diff(saved, wanted[name])
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
        return False, "none, will run"
    try:
        saved = json.loads(path.read_text()).get("qtool", {})
    except (OSError, json.JSONDecodeError):
        return False, "unreadable, will run"
    d = diff(saved, wanted, EVAL_FIELDS)
    if d:
        return False, "stale: " + "; ".join(d) + ", will run"
    return True, "valid, reused"


def render_plan(model: str, states: list[CheckpointState], schemes_: list[str], tasks: list[str], limit: int | None,
                baseline_note: str | None, no_eval: bool) -> list[str]:
    n = len(schemes_)
    out = [f"Plan for {model} ({n} scheme{'s' if n != 1 else ''})"]
    for st in states:
        if st.action in ("build", "rebuild"):
            users = ", ".join(st.schemes)
            out.append(f"  quantize  {st.name:<11} -> {users}" + ("  (--force)" if st.action == "rebuild" else ""))
        elif st.action == "reuse":
            out.append(f"  reuse     {st.name:<11} -> checkpoints/{st.name} (qtool.json matches)")
        else:
            out.append(f"  stale     {st.name:<11} -> checkpoints/{st.name} differs: " + "; ".join(st.diff))
    for s in schemes_:
        if schemes.SCHEMES[s].quantization:
            out.append(f"  runtime   {s:<11} (no checkpoint, vLLM quantizes to "
                       f"{schemes.SCHEMES[s].quantization.upper()} at load time)")
    if no_eval:
        out.append("  eval      skipped (--no-eval)")
    else:
        out.append(f"  eval      bf16 baseline ({baseline_note})")
        out.append(f"  eval      {', '.join(schemes_)}   tasks: {','.join(tasks)} ({_show(limit)})")
    return out
