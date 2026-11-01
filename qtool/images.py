# ABOUTME: Docker images qtool needs: tags derived from the Dockerfile hash (edit Dockerfile = new tag = rebuild),
# ABOUTME: compress variant by SM (cu126 has sm_70, cu129 has sm_75..sm_120), pinned versions and build commands.
import hashlib
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "images"
SCRIPTS = IMAGES / "scripts"


def dockerfile(kind: str) -> Path:
    return IMAGES / kind / "Dockerfile"


def _hash(kind: str) -> str:
    return hashlib.sha256(dockerfile(kind).read_bytes()).hexdigest()[:12]


def compress_cuda(cc: str) -> str:
    major, minor = (int(x) for x in cc.split("."))
    return "cu126" if (major, minor) < (7, 5) else "cu129"


def tag(kind: str, cuda: str | None = None) -> str:
    if kind == "compress":
        return f"qtool-compress:{cuda}-{_hash(kind)}"
    return f"qtool-{kind}:{_hash(kind)}"


def pinned(kind: str, package: str) -> str | None:
    m = re.search(rf"{re.escape(package)}==([\w.]+)", dockerfile(kind).read_text())
    return m[1] if m else None


def build_argv(kind: str, cuda: str | None = None) -> list[str]:
    args = ["--build-arg", f"CUDA={cuda}"] if kind == "compress" else []
    return ["docker", "build", "-t", tag(kind, cuda), *args, str(IMAGES / kind)]


def exists(image: str, run=subprocess.run) -> bool:
    return run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
