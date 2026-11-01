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


def counted_files(names: list[str], index_files: set[str] | None) -> list[str]:
    """Which *.safetensors to count: the index weight_map if present, else drop consolidated* duplicates."""
    if index_files is not None:
        return [n for n in names if n in index_files]
    plain = [n for n in names if not os.path.basename(n).startswith("consolidated")]
    return plain or names


def weights_info(model: str) -> tuple[int | None, int | None]:
    """(parameter count, bytes of *.safetensors) from local headers or Hub metadata; never downloads weights."""
    if os.path.isdir(model):
        root = Path(model)
        index = root / "model.safetensors.index.json"
        listed = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else None
        names = counted_files(sorted(f.name for f in root.glob("*.safetensors")), listed)
        if not names:
            return None, None
        return sum(header_params(root / n) for n in names), sum((root / n).stat().st_size for n in names)
    from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download

    info = HfApi().model_info(model, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors")}
    if not sizes:
        return None, None
    listed = None
    if "model.safetensors.index.json" in {s.rfilename for s in info.siblings}:
        path = hf_hub_download(model, "model.safetensors.index.json")
        listed = set(json.loads(Path(path).read_text())["weight_map"].values())
    names = counted_files(sorted(sizes), listed)
    meta = get_safetensors_metadata(model)
    params = sum(math.prod(t.shape) for n, fm in meta.files_metadata.items() if n in names
                 for t in fm.tensors.values())
    return params, sum(sizes[n] for n in names) or None


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
        tasks_missing = missing_tasks([t for t in a.tasks.split(",") if t]) if a.tasks else []
    except Exception as e:  # noqa: BLE001 - report the original error text to the host verbatim
        emit("probe_error", message=f"{type(e).__name__}: {e}")
        return EXIT_PROBE_ERROR
    qc = cfg.get("quantization_config") or None
    emit("probe", model=a.model, architectures=cfg.get("architectures"), model_type=cfg.get("model_type"),
         dtype=orig_dtype(cfg), multimodal=is_multimodal(cfg), remote_code=remote, quantized=bool(qc),
         quantization_config=qc, params=params, weight_bytes=size, chat_mode=mode, chat_reason=reason,
         tasks_missing=tasks_missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
