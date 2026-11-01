# ABOUTME: Container-tier tests for probe.py on local tiny model dirs: metadata, parameter count from
# ABOUTME: safetensors headers, chat-mode decision, remote-code and already-quantized detection.
import json
import shutil

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


def test_probe_ignores_consolidated_duplicate_weights(capsys, tiny_dir, tmp_path):
    src = tiny_dir("phi3")
    _, base = probe(capsys, src)
    d = tmp_path / "dup"
    shutil.copytree(src, d)
    shutil.copy(d / "model.safetensors", d / "consolidated.safetensors")
    _, e = probe(capsys, d)
    assert (e["params"], e["weight_bytes"]) == (base["params"], base["weight_bytes"])


def test_probe_counts_only_files_in_weight_map(capsys, tiny_dir, tmp_path):
    src = tiny_dir("phi3")
    _, base = probe(capsys, src)
    d = tmp_path / "idx"
    shutil.copytree(src, d)
    shutil.copy(d / "model.safetensors", d / "extra.safetensors")
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
    _, e = probe(capsys, d)
    assert (e["params"], e["weight_bytes"]) == (base["params"], base["weight_bytes"])


def test_probe_null_quantization_config_is_not_quantized(capsys, tiny_dir, tmp_path):
    d = tmp_path / "nullq"
    shutil.copytree(tiny_dir("phi3"), d)
    cfg = json.loads((d / "config.json").read_text())
    cfg["quantization_config"] = None
    (d / "config.json").write_text(json.dumps(cfg))
    _, e = probe(capsys, d)
    assert e["quantized"] is False
