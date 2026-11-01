# ABOUTME: Tests for qtool.console: duration/size formatting, header layout, rows and the live status line
# ABOUTME: on a TTY (rewritten in place) versus a pipe (throttled to one line per interval).
import io

import pytest

from qtool.console import WIDTH, Console, fmt_duration, fmt_gb


@pytest.mark.parametrize("sec,text", [(5, "5s"), (59.6, "1m 00s"), (527, "8m 47s"), (9605, "2h 40m")])
def test_fmt_duration(sec, text):
    assert fmt_duration(sec) == text


def test_fmt_gb():
    assert fmt_gb(5_200_000_000) == "5.2 GB"
    assert fmt_gb(None) == "unknown"


def make(tty, now=None):
    out = io.StringIO()
    clock = (lambda: now[0]) if now else None
    return out, Console(out, tty=tty, clock=clock or (lambda: 0.0), wallclock=lambda: "21:01:08")


def test_header_and_rows():
    out, c = make(False)
    c.header("[3/4] Quantizing INT4-W4A16")
    c.kv("recipe", "GPTQModifier")
    c.done(527)
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("══════ [3/4] Quantizing INT4-W4A16 ═") and lines[0].endswith("═ 21:01:08")
    assert len(lines[0]) == WIDTH
    assert lines[1] == "  recipe       : GPTQModifier"
    assert lines[2] == "  ✓ done in 8m 47s"


def test_header_with_custom_right_text():
    out, c = make(False)
    c.header("Summary · gsm8k strict-match (full)", right="total 41m 10s")
    assert out.getvalue().rstrip("\n").endswith("═ total 41m 10s")


def test_status_on_tty_rewrites_one_line():
    out, c = make(True)
    c.status("(1/33) Calibrating 10/512")
    c.status("(1/33) Calibrating 20/512")
    c.line("next")
    assert out.getvalue() == "\r\x1b[K  (1/33) Calibrating 10/512\r\x1b[K  (1/33) Calibrating 20/512\nnext\n"


def test_status_when_piped_is_throttled():
    now = [0.0]
    out, c = make(False, now)
    c.status("a")
    now[0] = 10.0
    c.status("b")
    now[0] = 31.0
    c.status("c")
    assert out.getvalue() == "  a\n  c\n"
