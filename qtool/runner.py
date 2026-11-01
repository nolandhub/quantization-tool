# ABOUTME: Runs containers: builds the `docker run` argv, streams merged stdout/stderr line by line (tqdm "\r" too),
# ABOUTME: appends the raw bytes to a log, calls a tick callback, stops the container on Ctrl-C; plus atomic dirs.
import codecs
import os
import re
import selectors
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContainerSpec:
    image: str
    command: list[str]
    name: str
    gpu: int | None = None
    env: dict = field(default_factory=dict)  # value None: pass the host variable through (-e NAME)
    mounts: list = field(default_factory=list)  # (host path, container path, read_only)
    memory_bytes: int | None = None


def docker_argv(spec: ContainerSpec) -> list[str]:
    argv = ["docker", "run", "--rm", "--name", spec.name]
    if spec.gpu is not None:
        argv += ["--gpus", f"device={spec.gpu}"]
    if spec.memory_bytes:
        argv += ["--memory", str(spec.memory_bytes), "--memory-swap", str(spec.memory_bytes)]
    for k, v in spec.env.items():
        argv += ["-e", k if v is None else f"{k}={v}"]
    for host, ctr, ro in spec.mounts:
        argv += ["-v", f"{host}:{ctr}" + (":ro" if ro else "")]
    return [*argv, spec.image, *spec.command]


class LineSplitter:
    def __init__(self):
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        parts = re.split(r"\r\n|\r|\n", self._buf + text)
        self._buf = parts.pop()
        return [p for p in parts if p]

    def flush(self) -> list[str]:
        rest, self._buf = self._buf, ""
        return [rest] if rest else []


def _docker_stop(name: str) -> None:
    subprocess.run(["docker", "stop", "-t", "10", name], capture_output=True)


def run_container(spec: ContainerSpec, log_path: Path, on_line, on_tick=None, popen=subprocess.Popen,
                  tick: float = 1.0, stop=_docker_stop) -> int:
    """Run to completion and return the exit code; on any exception stop the container, reap it and re-raise."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    splitter = LineSplitter()
    with open(log_path, "ab") as log:
        proc = popen(docker_argv(spec), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                if sel.select(timeout=tick):
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    for line in splitter.feed(decoder.decode(chunk)):
                        on_line(line)
                if on_tick:
                    on_tick()
            for line in splitter.feed(decoder.decode(b"", final=True)) + splitter.flush():
                on_line(line)
            return proc.wait()
        except BaseException:
            stop(spec.name)
            _drain(proc, sel, log)
            raise
        finally:
            sel.close()
            proc.stdout.close()


def _drain(proc, sel, log, deadline_s: float = 60.0) -> None:
    """Keep reading after `docker stop` so `docker run` is never blocked on a full pipe; kill it past the deadline."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if sel.select(timeout=1.0):
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            log.write(chunk)
    else:
        proc.kill()
    proc.wait()


@contextmanager
def atomic_dir(final: Path):
    """Yield <parent>/.tmp-<name>; rename to final only if the block succeeds, else delete it."""
    final = Path(final)
    tmp = final.with_name(".tmp-" + final.name)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
