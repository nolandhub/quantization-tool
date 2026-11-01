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


def test_int4_recipe_records_dampening():
    # The recipe text is part of the checkpoint signature: INT4 checkpoints made with llmcompressor's default
    # dampening (0.01) go stale. 0.05 matched RedHat on gsm8k (docs/spike/2026-11-01-acceptance-followup.md).
    assert "dampening_frac=0.05" in schemes.CHECKPOINTS["INT4-W4A16"]["recipe"]
    assert "dampening_frac" not in schemes.CHECKPOINTS["INT8-W8A8"]["recipe"]


@pytest.mark.parametrize("cc,col",[("7.0", "7.0"), ("6.1", "7.0"), ("7.5", "7.5"), ("8.0", "8.x"), ("8.6", "8.x"),
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
