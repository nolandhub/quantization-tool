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
