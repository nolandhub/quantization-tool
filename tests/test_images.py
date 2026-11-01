# ABOUTME: Tests for qtool.images: hash-based tags, cu126/cu129 choice by SM, versions pinned in the Dockerfiles
# ABOUTME: (eval image must carry num2words, compress image pytest) and the docker build command.
import re

import pytest

from qtool import images


def test_tags_follow_dockerfile_hash():
    assert re.fullmatch(r"qtool-compress:cu129-[0-9a-f]{12}", images.tag("compress", "cu129"))
    assert re.fullmatch(r"qtool-eval:[0-9a-f]{12}", images.tag("eval"))
    assert images.tag("compress", "cu126").split("-")[-1] == images.tag("compress", "cu129").split("-")[-1]


@pytest.mark.parametrize("cc,cuda", [("7.0", "cu126"), ("6.1", "cu126"), ("7.5", "cu129"), ("8.9", "cu129"), ("12.0", "cu129")])
def test_compress_cuda(cc, cuda):
    assert images.compress_cuda(cc) == cuda


def test_pins():
    assert images.pinned("compress", "llmcompressor") == "0.13.0"
    assert images.pinned("compress", "transformers") == "5.14.1"
    assert images.pinned("eval", "lm-eval[api]") == "0.4.13"
    assert "num2words" in images.dockerfile("eval").read_text()
    assert images.pinned("compress", "nope") is None


def test_build_argv():
    argv = images.build_argv("compress", "cu126")
    assert argv[:3] == ["docker", "build", "-t"] and argv[3] == images.tag("compress", "cu126")
    assert argv[4:6] == ["--build-arg", "CUDA=cu126"] and argv[-1].endswith("images/compress")
    assert images.build_argv("eval")[4].endswith("images/eval")


def test_exists_uses_docker_inspect():
    calls = []

    class R:
        returncode = 0

    assert images.exists("x:1", run=lambda argv, **kw: calls.append(argv) or R()) is True
    assert calls == [["docker", "image", "inspect", "x:1"]]
