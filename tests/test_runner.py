# ABOUTME: Tests for qtool.runner with a fake docker (a Python child replaying a real fixture log):
# ABOUTME: argv building, line splitting with "\r", raw log, exit codes, Ctrl-C stopping the container, atomic dirs.
import subprocess
import sys

import pytest

from qtool import runner
from qtool.parsers import QuantizeProgress


def spec(**over):
    return runner.ContainerSpec(**{"image": "img:1", "command": ["python", "/qtool/quantize.py"], "name": "qtool-1-0", **over})


def test_docker_argv_full():
    s = spec(gpu=1, memory_bytes=25_000_000_000, env={"HF_HOME": "/hf", "HF_TOKEN": None},
             mounts=[("/home/u/.cache/huggingface", "/hf", False), ("/repo/images/scripts", "/qtool", True)])
    assert runner.docker_argv(s) == [
        "docker", "run", "--rm", "--name", "qtool-1-0", "--gpus", "device=1",
        "--memory", "25000000000", "--memory-swap", "25000000000", "-e", "HF_HOME=/hf", "-e", "HF_TOKEN",
        "-v", "/home/u/.cache/huggingface:/hf", "-v", "/repo/images/scripts:/qtool:ro",
        "img:1", "python", "/qtool/quantize.py"]


def test_docker_argv_minimal():
    assert runner.docker_argv(spec()) == ["docker", "run", "--rm", "--name", "qtool-1-0", "img:1", "python",
                                          "/qtool/quantize.py"]


def test_line_splitter_across_chunks():
    s = runner.LineSplitter()
    assert s.feed("(1/33): Calib") == []
    assert s.feed("rating: 1/512\r(1/33): Calibrating: 2/512\r") == ["(1/33): Calibrating: 1/512", "(1/33): Calibrating: 2/512"]
    assert s.feed("\nlast") == []
    assert s.flush() == ["last"]


def fake_popen(log_file, code=0):
    script = (f"import sys; sys.stdout.buffer.write(open({str(log_file)!r}, 'rb').read()); "
              f"sys.stdout.flush(); sys.exit({code})")

    def popen(argv, **kw):
        assert argv[:2] == ["docker", "run"]
        return subprocess.Popen([sys.executable, "-c", script], **kw)

    return popen


def test_run_container_replays_real_log(fixtures, tmp_path):
    src = fixtures / "llmcompressor/w4a16-gptq.log"
    progress, ticks = QuantizeProgress(), []
    log = tmp_path / "logs" / "INT4-W4A16.quantize.log"
    code = runner.run_container(spec(), log, progress.feed, on_tick=lambda: ticks.append(1), popen=fake_popen(src))
    assert code == 0
    assert log.read_bytes() == src.read_bytes()
    assert (progress.subgraph, progress.total, progress.gptq_modules) == (33, 33, 128)
    assert ticks


def test_run_container_returns_exit_code(fixtures, tmp_path):
    code = runner.run_container(spec(), tmp_path / "x.log", lambda line: None,
                                popen=fake_popen(fixtures / "llmcompressor/w4a16-fp32-oom-killed.log", 137))
    assert code == 137


def test_ctrl_c_stops_container_and_reraises(fixtures, tmp_path):
    stopped = []

    def on_line(line):
        if "(2/33)" in line:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner.run_container(spec(name="qtool-7-3"), tmp_path / "x.log", on_line,
                             popen=fake_popen(fixtures / "llmcompressor/w4a16-gptq.log"), stop=stopped.append)
    assert stopped == ["qtool-7-3"]


def test_atomic_dir_renames_on_success_and_cleans_on_failure(tmp_path):
    final = tmp_path / "checkpoints" / "INT4-W4A16"
    with runner.atomic_dir(final) as tmp:
        assert tmp == tmp_path / "checkpoints" / ".tmp-INT4-W4A16"
        (tmp / "config.json").write_text("{}")
    assert (final / "config.json").exists() and not tmp.exists()
    with pytest.raises(RuntimeError):
        with runner.atomic_dir(final) as tmp:
            (tmp / "partial").write_text("x")
            raise RuntimeError("container failed")
    assert not tmp.exists() and (final / "config.json").exists()
