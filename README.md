# qtool

Make a Hugging Face model smaller (quantize it) and check how much accuracy it lost, with one command.

```bash
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16
```

qtool runs the 4 steps of the [vLLM LLM Compressor guide](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/)
for you: load the model, prepare calibration data, quantize, evaluate. It prints what each step is doing and ends
with a table: accuracy before and after, the drop in points, and the size on disk.

## What you need

- A Linux machine with an NVIDIA GPU (see [Which GPU for which scheme](#which-gpu-for-which-scheme)).
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
- Python 3.10 or newer. Nothing to `pip install`: qtool itself uses only the standard library.
- About 50 GB of free disk for the two Docker images, plus room for the models.
- For gated or private models: `export HF_TOKEN=...`.

## Install

```bash
git clone <this repo> qtool && cd qtool
ln -s "$PWD/bin/qtool" ~/.local/bin/qtool
qtool --version
```

The first run builds two Docker images (`qtool-compress` and `qtool-eval`). This takes a while once; later runs reuse them.

## Quick start

**1. See the plan without running anything:**

```bash
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16 --dry-run
```

**2. Quantize and evaluate:**

```bash
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16
```

On an RTX 4070 Ti SUPER (16 GB) this takes about 15 minutes to quantize a 4B model and about 10 minutes to evaluate
it on the full gsm8k set. Add `--eval-limit 50` for a quick check in a couple of minutes.

**3. Find the result:**

```
~/qtool/runs/Qwen3.5-4B/
├── checkpoints/INT4-W4A16/   the quantized model, ready for vLLM
├── results/                  accuracy per scheme + summary.json
└── logs/                     full logs of every step
```

Serve it like any model: `vllm serve ~/qtool/runs/Qwen3.5-4B/checkpoints/INT4-W4A16`.

## Common tasks

```bash
# Several schemes in one run (int4-w4a16 and int4-w4a8-fp8 share one checkpoint, so it is quantized once)
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16,int4-w4a8-fp8

# Only evaluate a model that already exists (any model vLLM can load), compared with another one
qtool eval RedHatAI/Qwen3.5-4B-quantized.w4a16 --baseline Qwen/Qwen3.5-4B

# Quantize only, evaluate later or on another GPU
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16 --no-eval

# Use your own calibration data (.jsonl with a 'messages' or 'text' field per line)
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16 --calib my-data.jsonl

# Machine with several GPUs: pick one
qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16 --gpu 1
```

`qtool quantize --help` and `qtool eval --help` list every option.

## Schemes

| Scheme | Weights / activations | What it is |
|---|---|---|
| `int4-w4a16` | 4-bit / 16-bit | GPTQ, smallest checkpoint, widest GPU support |
| `int4-w4a8-fp8` | 4-bit / FP8 | same checkpoint as `int4-w4a16`, faster kernel at runtime |
| `int4-w4a8-int8` | 4-bit / INT8 | same checkpoint as `int4-w4a16`, faster kernel at runtime |
| `int8-w8a8` | 8-bit / 8-bit | SmoothQuant + GPTQ |
| `fp8-w8a8` | FP8 / FP8 | vLLM quantizes on the fly while loading: no checkpoint, nothing to quantize |

If you are not sure, start with `int4-w4a16`.

## Which GPU for which scheme

| GPU generation (compute capability) | Examples | Quantize | Evaluate |
|---|---|---|---|
| Volta (7.0) | V100 | yes (float32, about 10× slower) | no: vLLM 0.28 has no kernels |
| Turing (7.5) | T4, RTX 20xx | yes (float32) | `int8-w8a8` |
| Ampere (8.0, 8.6) | A100, RTX 30xx | yes | `int4-w4a16`, `int8-w8a8`, `fp8-w8a8` (as weight-only) |
| Ada (8.9) | L4, L40S, RTX 40xx | yes | all schemes |
| Hopper (9.0) | H100, H200 | yes | all except `int4-w4a8-*` |
| Blackwell (10.x, 12.x) | B200, RTX 50xx | yes | `int4-w4a16`, `fp8-w8a8` (+ `int4-w4a8-*` on 12.x) |

qtool checks this before it starts. If a scheme cannot run on your GPU it says why and suggests one that can.

## When something goes wrong

| You see | What to do |
|---|---|
| `error: checkpoints/... was made with different settings` | You changed an option since the last run. Add `--force` to quantize again. |
| `another qtool run (pid N) is active` | Only one qtool run at a time per user. Wait for it, or stop it. |
| `gated or private model: export HF_TOKEN` | Accept the model's license on Hugging Face, then `export HF_TOKEN=...`. |
| `... does not fit` / `host RAM ran out` | The model is too big for this GPU or this machine's RAM. Use a bigger machine or a smaller model. |
| A scheme shows `FAILED` in the summary | The reason is printed above the table; the full log is in `~/qtool/runs/<model>/logs/`. |

Exit codes: `0` all good, `1` at least one scheme failed, `2` wrong option or the machine cannot run it, `130` stopped with Ctrl-C.

qtool never stops or starts other containers or services. If something else is using the GPU, it tells you and lets
you decide.

## Good to know

- Running the same command twice reuses the checkpoint and the cached baseline score, so the second run is fast.
- Quantizing the same model twice does not give exactly the same checkpoint: GPTQ on a GPU is not bit-for-bit
  repeatable, and gsm8k scores of separate runs can differ by about ±2 points.
  Details: `docs/spike/2026-11-01-acceptance-followup.md`.

## For developers

```bash
uv run --no-project --with pytest python -m pytest     # host tests: no GPU, no Docker needed
scripts/test-container.sh                              # container tests: run on a machine with Docker
```

- Design: `docs/specs/2026-10-30-qtool-design.md`
- Implementation plan: `docs/plans/2026-11-01-qtool-implementation.md`
- Experiment results: `docs/spike/`
- `sync.sh` copies the checkout to the team's GPU test machine (rtx4070). Adjust the host name for your own setup.
