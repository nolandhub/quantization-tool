# NVFP4 acceptance — 2026-11-02

Spec 3.2 §9 acceptance for `nvfp4-w4a4`. Machine: rtx4070 (RTX 4070 Ti SUPER 16 GB, SM 8.9), images
`qtool-compress:cu129-4d4b1be4056f` / `qtool-eval:61d6a2cbbc15`, model `Qwen/Qwen3.5-4B`, gsm8k full (1319), chat mode.
Logs: `rtx4070:~/qtool/acc-a-nvfp4.log`, `acc-b-axion.log`; fixtures `tests/fixtures/lm_eval/eval-nvfp4.log`,
`tests/fixtures/llmcompressor/nvfp4-qmod.log`.

## (a) `qtool quantize Qwen/Qwen3.5-4B --scheme nvfp4-w4a4`

- Preflight: `nvfp4-w4a4 on SM 8.9 runs as weight-only FP4 (Marlin), activations stay BF16`.
- Recipe `QuantizationModifier targets=Linear scheme=NVFP4 · sequential Qwen3_5DecoderLayer`; ignore lm_head, visual,
  linear_attn; 128 Linear quantized (same set as INT4-W4A16). Progress `(N/33) Calibrating` parsed as for GPTQ.
- Step 3: 8 min 49 s (INT4 GPTQ: ~15 min). Checkpoint 9.3 GB → 5.7 GB, MTP saved.
- vLLM: `Using MarlinNvFp4LinearKernel for NVFP4 GEMM`, model 5.23 GiB, KV 103,277 tokens.

```
scheme          status   accuracy        Δ vs base   recovery   size
bf16            OK       0.805 ± 0.011   —           —          9.3 GB
nvfp4-w4a4      OK       0.804 ± 0.011   −0.1 pp     99.9%      5.7 GB
```

## (b) `qtool eval AxionML/Qwen3.5-4B-NVFP4 --baseline Qwen/Qwen3.5-4B` (reference, used in the bake-off)

```
scheme            status   accuracy        Δ vs base   recovery   size
baseline          OK       0.810 ± 0.011   —           —          9.3 GB
Qwen3.5-4B-NVFP4  OK       0.763 ± 0.012   −4.7 pp     94.2%      4.2 GB
```

- Format `modelopt · NVFP4 W4A4 g16` (NVIDIA ModelOpt, not llm-compressor); same Marlin kernel.
- Its ignore list is only `lm_head` + `linear_attn.conv1d`: it also quantizes the Gated DeltaNet Linears that qtool keeps
  in BF16 → 1.5 GB smaller, 4.6 pp lower than qtool's NVFP4 here. Possible cause of the low NVFP4 accuracy in the
  bake-off (same checkpoint) — hypothesis, not tested.

## Observations

- BF16 baseline, same config, two runs: 0.8052 and 0.8097 (0.45 pp apart). vLLM eval is not bit-for-bit repeatable;
  compare each scheme with the baseline of its own run (Δ column).
- On SM 8.9 NVFP4 is weight-only, so it is not faster than INT4-W4A16 and is ~10 % larger; its value is on Blackwell
  (FP4×FP4), which remains unverified.

## Skipped (decided with Nhân)

- (c) GPTQ-NVFP4: the rule was "switch the default to GPTQ if it beats RTN by > 2 pp". RTN already recovers 99.9 %
  (−0.1 pp), so GPTQ cannot clear the bar. Default stays `QuantizationModifier`.
- (d) `--scheme int4-w4a8-fp8`: code path identical to `main` (unit-tested); a re-run needs a fresh INT4 quantize
  (~25 min) for no new information.
