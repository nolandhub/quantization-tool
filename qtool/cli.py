# ABOUTME: Command-line interface: parsing and validation for `qtool quantize` / `qtool eval`, preflight wiring
# ABOUTME: (GPU, images, probe, thresholds) and exit codes: 0 ok, 1 a scheme failed, 2 usage/preflight error, 130 Ctrl-C.
import argparse

from qtool import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qtool", description="Quantize a model with LLM Compressor and evaluate it in vLLM.")
    p.add_argument("--version", action="version", version=f"qtool {__version__}")
    return p


def main(argv=None) -> int:
    build_parser().parse_args(argv)
    return 0
