# qtool

Quantize a Hugging Face model with LLM Compressor and evaluate it in vLLM — the 4 steps of the
[vLLM LLM Compressor guide](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/)
(load, calibrate, quantize, evaluate) as one command with step-by-step console output.

Design: `docs/specs/2026-10-30-qtool-design.md` · spike results: `docs/spike/2026-10-30-findings.md`.

## Requirements

- NVIDIA GPU, Docker with NVIDIA Container Toolkit, Python ≥ 3.10 on the host (standard library only).
- Images are built on first use: `qtool-compress` (torch 2.13, llmcompressor 0.13.0, transformers 5.14.1;
  `cu126` variant for SM < 7.5) and `qtool-eval` (vLLM 0.28 + lm-eval 0.4.13).

## Install

```bash
ln -s "$PWD/bin/qtool" ~/.local/bin/qtool
```

## Use

```bash
# quantize into two schemes sharing one INT4 checkpoint, then evaluate both against BF16 on gsm8k
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16,int4-w4a8-fp8

# only evaluate (any model vLLM can load), with a cached baseline for Δ pp / recovery %
qtool eval RedHatAI/Qwen3.5-4B-quantized.w4a16 --act fp8 --baseline Qwen/Qwen3.5-4B

# V100 (SM 7.0): quantize only, evaluate the checkpoint on another GPU later
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16 --no-eval --gpu 0
```

Schemes: `int4-w4a16`, `int4-w4a8-fp8`, `int4-w4a8-int8` (INT4 checkpoint + Marlin W4A8 at runtime, SM 8.9 / 12.0),
`int8-w8a8` (SmoothQuant + GPTQ), `fp8-w8a8` (vLLM online FP8, no checkpoint).

Results live in `~/qtool/runs/<model>/` (`checkpoints/`, `results/`, `logs/`). qtool never stops or starts other
containers or services: preflight only reports what is using the GPU.

## Develop

```bash
uv run --no-project --with pytest python -m pytest      # host tests (no GPU, no Docker)
./sync.sh push && ssh rtx4070 'cd ~/qtool && scripts/test-container.sh'   # container tests (CPU, compress image)
```

GPU integration and acceptance runs are manual (see `docs/plans/2026-11-01-qtool-implementation.md`, tasks 16–17).
