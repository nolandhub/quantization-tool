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
# GPTQ dampening: 0.05 for W4A16 matched RedHat on gsm8k (docs/spike/2026-11-01-acceptance-followup.md);
# W8A8 keeps llmcompressor's default 0.01 (no evidence either way)
DAMPENING = {"W4A16": 0.05, "W8A8": 0.01}
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
    ap.add_argument("--signature", default="{}", type=json.loads, help="JSON written verbatim into qtool.json for cache checks")
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


def cast_saved_tensors(out: Path, dtype: str) -> int:
    """Rewrite float32 tensors in out/*.safetensors as `dtype` (spike Q6); returns how many were converted.

    Done on the saved files because llmcompressor's dispatch/offload defeats an in-memory model.to() on GPU.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    target, converted, total_size = getattr(torch, dtype), 0, 0
    for path in sorted(out.glob("*.safetensors")):
        with safe_open(path, "pt") as f:
            metadata = f.metadata() or {"format": "pt"}
        tensors = load_file(path)
        for k, t in tensors.items():
            if t.dtype == torch.float32:
                tensors[k] = t.to(target)
                converted += 1
        save_file(tensors, path, metadata=metadata)
        total_size += sum(t.nbytes for t in tensors.values())

    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    for c in (cfg, cfg.get("text_config")):
        if isinstance(c, dict):
            for key in ("dtype", "torch_dtype"):
                if key in c:
                    c[key] = dtype
    cfg_path.write_text(json.dumps(cfg, indent=2))

    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index.setdefault("metadata", {})["total_size"] = total_size
        index_path.write_text(json.dumps(index, indent=2))
    return converted


def main(argv=None) -> int:
    a = parse_args(argv)
    import torch
    import transformers

    emit("step", n=1)
    cfg, _ = transformers.PretrainedConfig.get_config_dict(a.model)
    qc = cfg.get("quantization_config")
    if qc:
        method = qc.get("quant_method", "unknown")
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
    if r["linears_quantized"] == 0:
        emit("refused", reason="derived ignore list excludes every Linear (nothing to quantize)",
             hint="check --ignore and the architecture")
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
    gptq = GPTQModifier(targets="Linear", scheme=preset, ignore=r["ignore"], dampening_frac=DAMPENING[preset])
    recipe = [SmoothQuantModifier(smoothing_strength=0.8), gptq] if preset == "W8A8" else gptq
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    oneshot(model=model, processor=tok, dataset=ds, recipe=recipe, max_seq_length=a.seq_len,
            num_calibration_samples=len(ds), sequential_targets=[r["sequential_target"]])
    oneshot_seconds = round(time.time() - t0, 1)
    emit("oneshot", seconds=oneshot_seconds, peak_gpu_bytes=torch.cuda.max_memory_allocated() if cuda else None)

    cast, cast_tensors = None, 0
    orig = orig_dtype(cfg)
    out = Path(a.out)
    model.save_pretrained(out, save_compressed=True)
    if orig in ("bfloat16", "float16") and orig != a.compute_dtype:
        cast = f"{a.compute_dtype}->{orig}"  # float32 compute: store unquantized tensors in the source dtype (Q6)
        cast_tensors = cast_saved_tensors(out, orig)
    tok.save_pretrained(out)
    copied = copy_side_files(a.model, out)
    if r["mtp"]:
        from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
        save_mtp_tensors_to_checkpoint(source_model=a.model, dest_dir=str(out))
    import compressed_tensors
    import llmcompressor

    (out / "qtool.json").write_text(json.dumps({
        "signature": a.signature,
        "recipe": {k: r[k] for k in ("ignore", "sequential_target", "num_layers", "linears_quantized")},
        "calib": {k: v for k, v in stats.items() if k != "cached"},
        "compute_dtype": a.compute_dtype, "cast": cast, "oneshot_seconds": oneshot_seconds,
        "versions": {"llmcompressor": llmcompressor.__version__, "transformers": transformers.__version__,
                     "torch": torch.__version__, "compressed_tensors": compressed_tensors.__version__},
    }, indent=1))
    emit("saved", path=str(out), bytes=dir_bytes(out), cast=cast, cast_tensors=cast_tensors, mtp=r["mtp"], copied=copied)
    return 0


if __name__ == "__main__":
    sys.exit(main())
