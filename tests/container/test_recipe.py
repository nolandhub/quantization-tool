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
