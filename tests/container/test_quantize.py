# ABOUTME: Container-tier end-to-end tests for quantize.py on tiny models (CPU): events, checkpoint format,
# ABOUTME: float32-compute cast back to the source dtype, calibration cache, refusal of already-quantized models.
import json
import shutil
import struct

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


def tensor_dtypes(directory):
    found = set()
    for f in directory.glob("*.safetensors"):
        with open(f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        found |= {v["dtype"] for k, v in header.items() if k != "__metadata__"}
    return found


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
    assert tensor_dtypes(out) and "F32" not in tensor_dtypes(out)
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


def test_empty_recipe_is_refused(capsys, tiny_dir, calib_jsonl, tmp_path):
    code, evs = run(capsys, tiny_dir("phi3"), tmp_path / "out", calib_jsonl, tmp_path / "calib",
                    extra=["--ignore", "re:.*"])
    assert code == 3
    assert evs[-1]["event"] == "refused" and "nothing to quantize" in evs[-1]["reason"]
    assert [e["n"] for e in evs if e["event"] == "step"] == [1]


def test_side_files_copied_but_weight_index_is_not(capsys, tiny_dir, calib_jsonl, tmp_path):
    src = tmp_path / "src"
    shutil.copytree(tiny_dir("phi3"), src)
    (src / "preprocessor_config.json").write_text("{}")
    (src / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    out = tmp_path / "out"
    code, evs = run(capsys, src, out, calib_jsonl, tmp_path / "calib")
    assert code == 0
    assert by_name(evs)["saved"]["copied"] == ["preprocessor_config.json"]
    assert not (out / "model.safetensors.index.json").exists()


def test_null_quantization_config_is_not_refused(capsys, tiny_dir, calib_jsonl, tmp_path):
    src = tmp_path / "src"
    shutil.copytree(tiny_dir("phi3"), src)
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = None
    (src / "config.json").write_text(json.dumps(cfg))
    code, _ = run(capsys, src, tmp_path / "out", calib_jsonl, tmp_path / "calib")
    assert code == 0
