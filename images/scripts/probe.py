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
