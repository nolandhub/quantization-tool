# ABOUTME: Console rendering: step headers, key/value rows, done/fail marks, durations, sizes and a live status line.
# ABOUTME: The status line is rewritten in place on a TTY; when piped, at most one status line per interval is printed.
import sys
import time

WIDTH = 64


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def fmt_gb(nbytes: float | None) -> str:
    return "unknown" if nbytes is None else f"{nbytes / 1e9:.1f} GB"


class Console:
    def __init__(self, stream=None, tty: bool | None = None, clock=time.monotonic, wallclock=None,
                 status_interval: float = 30.0):
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty() if tty is None else tty
        self.clock = clock
        self.wallclock = wallclock or (lambda: time.strftime("%H:%M:%S"))
        self.status_interval = status_interval
        self._status_shown = False
        self._last_status_at: float | None = None

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def line(self, text: str = "") -> None:
        self.end_status()
        self._write(text + "\n")

    def header(self, title: str, right: str | None = None) -> None:
        left, right = f"══════ {title} ", f" {self.wallclock() if right is None else right}"
        self.line(left + "═" * max(3, WIDTH - len(left) - len(right)) + right)

    def kv(self, key: str, value) -> None:
        self.line(f"  {key:<13}: {value}")

    def done(self, seconds: float) -> None:
        self.line(f"  ✓ done in {fmt_duration(seconds)}")

    def failed(self, reason: str) -> None:
        self.line(f"  ✗ FAILED: {reason}")

    def warn(self, text: str) -> None:
        self.line(f"  ! {text}")

    def status(self, text: str) -> None:
        if self.tty:
            self._write("\r\x1b[K  " + text)
            self._status_shown = True
            return
        now = self.clock()
        if self._last_status_at is None or now - self._last_status_at >= self.status_interval:
            self._last_status_at = now
            self._write("  " + text + "\n")

    def end_status(self) -> None:
        if self._status_shown:
            self._write("\n")
            self._status_shown = False
        self._last_status_at = None
