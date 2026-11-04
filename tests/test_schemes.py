# ABOUTME: Tests for qtool.schemes: scheme parsing, checkpoint sharing, SM capability matrix, compute dtype.
# ABOUTME: Matrix expectations are copied from spec §3.
import pytest

from qtool import schemes


def test_parse_schemes_dedupes_and_keeps_order():
    assert schemes.parse_schemes("int4-w4a8-fp8, int4-w4a16,int4-w4a8-fp8") == ["int4-w4a8-fp8", "int4-w4a16"]


def test_parse_schemes_rejects_unknown_and_lists_groups():
    with pytest.raises(ValueError, match="unknown scheme 'int4-w4a8'") as e:
        schemes.parse_schemes("int4-w4a8")
    assert str(e.value) == "unknown scheme 'int4-w4a8'.\n" + schemes.scheme_help()


def test_scheme_help_lists_the_two_groups():
    assert schemes.scheme_help() == (
        "schemes that build a checkpoint:\n"
        "  int4-w4a16      INT4 weights, 16-bit activations (checkpoint INT4-W4A16)\n"
        "  int8-w8a8       INT8 weights and activations (checkpoint INT8-W8A8)\n"
        "  nvfp4-w4a4      FP4 weights and activations (checkpoint NVFP4)\n"
        "  fp8-w8a8        FP8 weights, FP8 activations at runtime (checkpoint FP8-DYNAMIC)\n"
        "schemes that reuse INT4-W4A16 (quantized once, no extra checkpoint):\n"
        "  int4-w4a8-fp8   INT4 weights, FP8 activations at runtime (Marlin)\n"
        "  int4-w4a8-int8  INT4 weights, INT8 activations at runtime (Marlin)")


def test_groups_cover_exactly_the_schemes_in_order():
    grouped = [name for names in schemes.GROUPS.values() for name in names]
    assert sorted(grouped) == sorted(schemes.SCHEMES) and len(schemes.SCHEMES) == 6
    assert list(schemes.SCHEMES) == ["int4-w4a16", "int4-w4a8-fp8", "int4-w4a8-int8", "int8-w8a8", "nvfp4-w4a4", "fp8-w8a8"]


def test_parse_schemes_rejects_empty():
    with pytest.raises(ValueError, match="--scheme is empty"):
        schemes.parse_schemes(" , ")


def test_checkpoints_are_shared_and_fp8_has_its_own():
    got = schemes.checkpoints_for(["int4-w4a8-fp8", "int4-w4a16", "fp8-w8a8", "int8-w8a8"])
    assert got == {"INT4-W4A16": ["int4-w4a8-fp8", "int4-w4a16"], "FP8-DYNAMIC": ["fp8-w8a8"], "INT8-W8A8": ["int8-w8a8"]}


def test_only_fp8_dynamic_is_data_free():
    assert [c for c in schemes.CHECKPOINTS if schemes.data_free(c)] == ["FP8-DYNAMIC"]
    assert schemes.CHECKPOINTS["FP8-DYNAMIC"]["recipe"] == "QuantizationModifier targets=Linear scheme=FP8_DYNAMIC"


def test_nvfp4_has_its_own_checkpoint():
    assert schemes.checkpoints_for(["nvfp4-w4a4", "int4-w4a8-int8"]) == \
        {"NVFP4": ["nvfp4-w4a4"], "INT4-W4A16": ["int4-w4a8-int8"]}
    assert schemes.SCHEMES["int4-w4a8-int8"].env == {"VLLM_MARLIN_INPUT_DTYPE": "int8"}


def test_int4_recipe_records_dampening():
    # The recipe text is part of the checkpoint signature: INT4 checkpoints made with llmcompressor's default
    # dampening (0.01) go stale. 0.05 matches RedHat's published recipe (docs/spike/2026-11-01-acceptance-followup.md).
    assert "dampening_frac=0.05" in schemes.CHECKPOINTS["INT4-W4A16"]["recipe"]
    assert "dampening_frac" not in schemes.CHECKPOINTS["INT8-W8A8"]["recipe"]


@pytest.mark.parametrize("cc,col",[("7.0", "7.0"), ("6.1", "7.0"), ("7.5", "7.5"), ("8.0", "8.x"), ("8.6", "8.x"),
                                    ("8.9", "8.9"), ("9.0", "9.0"), ("10.0", "10.x"), ("12.0", "12.x")])
def test_sm_column(cc, col):
    assert schemes.sm_column(cc) == col


def test_support_matrix_samples():
    assert schemes.support("int4-w4a8", "8.9") == schemes.Support("ok", "", True)
    nv = schemes.support("nvfp4-w4a4", "8.9")
    assert (nv.status, nv.verified) == ("warn", True)
    assert "weight-only FP4" in nv.note
    nv12 = schemes.support("nvfp4-w4a4", "12.0")
    assert (nv12.status, nv12.verified) == ("ok", False)
    assert schemes.support("nvfp4-w4a4", "7.0").status == "no"
    assert schemes.support("int8-w8a8", "12.0").status == "no"
    fp8 = schemes.support("fp8-w8a8", "8.6")
    assert (fp8.status, fp8.note, fp8.verified) == ("warn", "runs as W8A16 weight-only (Marlin)", False)
    assert schemes.support("quantize", "7.0") == schemes.Support("warn", "float32 compute, about 10x slower than bfloat16", True)
    assert schemes.support("baseline", "7.0").status == "no"


def test_alternative_prefers_same_width():
    assert schemes.alternative("int8-w8a8", "10.0") == "fp8-w8a8"
    assert schemes.alternative("int4-w4a8-fp8", "9.0") == "int4-w4a16"
    assert schemes.alternative("nvfp4-w4a4", "7.0") is None
    assert schemes.alternative("nvfp4-w4a4", "8.9") == "int4-w4a16"
    assert schemes.alternative("int4-w4a16", "7.0") is None


@pytest.mark.parametrize("cc,req,want", [("8.9", "auto", "bfloat16"), ("7.0", "auto", "float32"),
                                         ("7.5", "auto", "float32"), ("7.0", "bfloat16", "bfloat16")])
def test_compute_dtype(cc, req, want):
    assert schemes.compute_dtype(cc, req) == want
