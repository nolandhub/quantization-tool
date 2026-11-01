# ABOUTME: Tests for qtool.report: Δ pp, recovery %, the summary table (spec §8 layout) and summary.json.
# ABOUTME: Baseline missing or skipped must show n/a instead of a made-up comparison.
import json

import pytest

from qtool import report
from qtool.parsers import Metric


def gsm(v, se=0.011):
    return {"gsm8k": Metric("gsm8k", "exact_match", "strict-match", v, se)}


def test_delta_and_recovery():
    assert report.delta_pp(0.792, 0.812) == pytest.approx(-2.0)
    assert report.recovery(0.792, 0.812) == pytest.approx(97.536, abs=1e-3)
    assert report.recovery(0.5, 0.0) is None


def test_summary_table_matches_spec_layout():
    rows = [report.Row("bf16", "OK", gsm(0.812), 8_800_000_000),
            report.Row("int4-w4a16", "OK", gsm(0.792), 5_200_000_000),
            report.Row("int4-w4a8-fp8", "OK", gsm(0.784), 5_200_000_000, "shared"),
            report.Row("int8-w8a8", "FAILED")]
    assert report.summary_title("gsm8k", rows, None) == "Summary · gsm8k strict-match (full)"
    assert report.summary_lines(rows, "gsm8k", "bf16") == [
        "  scheme          status   accuracy        Δ vs base   recovery   size",
        "  bf16            OK       0.812 ± 0.011   —           —          8.8 GB",
        "  int4-w4a16      OK       0.792 ± 0.011   −2.0 pp     97.5%      5.2 GB",
        "  int4-w4a8-fp8   OK       0.784 ± 0.011   −2.8 pp     96.6%      5.2 GB (shared)",
        "  int8-w8a8       FAILED   —               n/a         n/a        unknown",
    ]


def test_skipped_baseline_gives_na():
    rows = [report.Row("bf16", "SKIPPED", note="does not fit"), report.Row("int4-w4a16", "OK", gsm(0.68, 0.067))]
    assert report.summary_lines(rows, "gsm8k", "bf16")[2] == \
        "  int4-w4a16      OK       0.680 ± 0.067   n/a         n/a        unknown"
    assert report.summary_title("gsm8k", rows, 50) == "Summary · gsm8k strict-match (limit 50)"


def test_long_labels_widen_the_first_column():
    rows = [report.Row("baseline", "OK", gsm(0.70, 0.105), 9_300_000_000),
            report.Row("INT4-W4A16+act-fp8", "OK", gsm(0.75, 0.099), 5_500_000_000)]
    assert report.summary_lines(rows, "gsm8k", "baseline") == [
        "  scheme              status   accuracy        Δ vs base   recovery   size",
        "  baseline            OK       0.700 ± 0.105   —           —          9.3 GB",
        "  INT4-W4A16+act-fp8  OK       0.750 ± 0.099   +5.0 pp     107.1%     5.5 GB",
    ]


def test_write_summary(tmp_path):
    path = tmp_path / "results" / "summary.json"
    report.write_summary(path, [report.Row("bf16", "OK", gsm(0.8), 1)], {"model": "m"})
    data = json.loads(path.read_text())
    assert data["meta"] == {"model": "m"}
    assert data["rows"][0]["metrics"]["gsm8k"]["value"] == 0.8 and data["rows"][0]["status"] == "OK"
