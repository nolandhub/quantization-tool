# qtool Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trả lời 8 câu hỏi của spike (spec §10) bằng chạy thật, thu fixture log thật, ghi kết quả vào `docs/spike/2026-10-30-findings.md` để viết plan implementation (plan 2).

**Architecture:** Script spike tạm chạy trong 2 image Docker trên rtx4070 (`~/qtool-spike/`, **không commit**). Chỉ commit vào repo: file findings, Dockerfile đã chốt phiên bản, và fixture trong `tests/fixtures/`.

**Tech Stack:** Docker, `vllm/vllm-openai:v0.28.0-cu129-ubuntu2404`, PyTorch image CUDA 12.x, llmcompressor, transformers, datasets, lm-eval.

**Spec:** `docs/specs/2026-10-30-qtool-design.md` (§10 Spike; §3 bảng phần cứng; §5 kiến trúc)

**Plan 2 (implementation) chỉ viết sau khi spike xong**, vì regex parser, ignore list, phiên bản ghim, ngưỡng VRAM và chế độ eval đều lấy từ kết quả spike.

## Global Constraints

- Image eval gốc: `vllm/vllm-openai:v0.28.0-cu129-ubuntu2404` (đã có trên rtx4070).
- `lm-eval[api]>=0.4.12` (docs vLLM).
- Calib mặc định: `HuggingFaceH4/ultrachat_200k` split `train_sft`, 512 mẫu, seq len 2048, shuffle seed 42, áp chat template.
- Model thử: `Qwen/Qwen3.5-4B` (BF16). Checkpoint đối chứng: `RedHatAI/Qwen3.5-4B-quantized.w4a16` (đã có trong HF cache rtx4070).
- HF cache rtx4070: `~/.cache/huggingface` (không phải `~/hf-cache`).
- Spike **không** dừng / bật container nào. Trước mỗi task chạy GPU: kiểm `nvidia-smi`; GPU bận → dừng, báo Nhân.
- Apollo: **hỏi Nhân dùng GPU nào trước khi chạy**; image/docker data trên Apollo đặt ở `/u01` (đĩa gốc chậm 7–20 MB/s).
- Mọi file code có 2 dòng `# ABOUTME:` ở đầu. Commit không có dòng attribution AI.
- Không đoán: câu nào spike không trả lời được → ghi "chưa trả lời được" + lý do trong findings.

## Đã biết trước spike (từ checkpoint RedHat trên rtx4070, 30/10)

Đọc từ `RedHatAI/Qwen3.5-4B-quantized.w4a16`: `recipe.yaml`, `config.json`, `README.md`:

- Kiến trúc: `Qwen3_5ForConditionalGeneration` (có vision encoder + MTP). README load bằng
  `Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")`.
- Recipe RedHat: `GPTQModifier(targets="Linear", scheme="W4A16", sequential_targets=["Qwen3_5DecoderLayer"], dampening_frac=0.05)`,
  `actorder: static`, `block_size: 128`.
- Ignore RedHat: `re:.*lm_head`, `re:.*embed_tokens$`, `re:.*visual.*`, `re:.*model.visual.*`, `re:.*linear_attn.*` (lớp GDN không quantize); `config.json` còn có `re:^mtp.*`.
- RedHat calib: Open-Platypus 1024 mẫu, seq 8192 (khác docs vLLM: ultrachat 512 / 2048).
- RedHat lưu MTP bằng `save_mtp_tensors_to_checkpoint(source_model=..., dest_dir=...)` (README không ghi import từ đâu).
- RedHat eval: `lm_eval --model local-chat-completions --apply_chat_template`, thinking bật, `max_gen_toks=65536`, task `gsm8k_platinum_cot_llama` 0-shot, 3 seed; vLLM serve `--language-model-only`.

---

### Task 1: Chuẩn bị thư mục spike, tải model gốc, fixture nvidia-smi

**Files:**
- Create (rtx4070, không commit): `~/qtool-spike/`
- Create (repo): `tests/fixtures/nvidia-smi/rtx4070.csv`, `tests/fixtures/nvidia-smi/rtx4070-apps.csv`

- [ ] **Step 1: Kiểm GPU rảnh**

Run: `ssh rtx4070 'nvidia-smi --query-gpu=index,name,compute_cap,memory.used,memory.total --format=csv'`
Expected: `memory.used` < 500 MiB. Nếu không → dừng, báo Nhân.

- [ ] **Step 2: Tạo thư mục và tải model gốc**

```bash
ssh rtx4070 'mkdir -p ~/qtool-spike/{out,logs,ckpt} && \
  docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface --entrypoint python3 \
  vllm/vllm-openai:v0.28.0-cu129-ubuntu2404 -c "from huggingface_hub import snapshot_download as d; print(d(\"Qwen/Qwen3.5-4B\"))" 2>&1 | tail -3'
```
Expected: in đường dẫn snapshot. Ghi lại thời gian tải.

- [ ] **Step 3: Đọc config model gốc**

```bash
ssh rtx4070 'S=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/); python3 -c "
import json,sys; c=json.load(open(sys.argv[1]+\"config.json\"))
print(c.get(\"architectures\"), c.get(\"torch_dtype\") or c.get(\"dtype\"), \"quantization_config\" in c)
t=c.get(\"text_config\",c); print(t.get(\"num_hidden_layers\"), t.get(\"layer_types\"))" $S'
```
Expected: `['Qwen3_5ForConditionalGeneration'] bfloat16 False`, số layer và `layer_types`. Ghi vào findings Q2.

- [ ] **Step 4: Thu fixture nvidia-smi** (dùng cho test preflight ở plan 2)

```bash
mkdir -p tests/fixtures/nvidia-smi
ssh rtx4070 'nvidia-smi --query-gpu=index,name,compute_cap,memory.used,memory.total,memory.free --format=csv,noheader,nounits' > tests/fixtures/nvidia-smi/rtx4070.csv
ssh rtx4070 'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits' > tests/fixtures/nvidia-smi/rtx4070-apps.csv
ssh rtx4070 'nvidia-smi --query-gpu=index,uuid --format=csv,noheader' > tests/fixtures/nvidia-smi/rtx4070-uuid.csv
```

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/nvidia-smi && git commit -m "test: add rtx4070 nvidia-smi fixtures"
```

---

### Task 2: Image compress — tìm tổ hợp phiên bản

Trả lời findings Q1 (một nửa) và Q6 (một nửa: torch có SM 7.0).

**Files:**
- Create (repo): `images/compress/Dockerfile`

- [ ] **Step 1: Viết Dockerfile ứng viên**

```dockerfile
# ABOUTME: Image cho step 1-3 cua qtool: transformers + llmcompressor tren torch CUDA 12.x.
# ABOUTME: Torch CUDA 12.x de con kernel SM 7.0 (V100); phien ban ghim theo spike 2026-10-30.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir llmcompressor datasets
```

- [ ] **Step 2: Build và ghi lại phiên bản pip đã chọn**

```bash
rsync -a images/ rtx4070:qtool-spike/images/
ssh rtx4070 'cd ~/qtool-spike && docker build -t qtool-compress:spike images/compress 2>&1 | tail -5 && \
  docker run --rm --gpus all qtool-compress:spike python -c "
import torch, transformers, llmcompressor, compressed_tensors
print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())
print(transformers.__version__, llmcompressor.__version__, compressed_tensors.__version__)
from transformers import Qwen3_5ForConditionalGeneration; print(\"Qwen3_5 OK\")"'
```
Expected: in phiên bản; `get_arch_list()` có `sm_70`; dòng `Qwen3_5 OK`.

- [ ] **Step 3: Xử lý theo kết quả**

| Kết quả | Làm gì |
|---|---|
| pip nâng torch lên bản khác 2.8 | Ghi lại bản torch; kiểm `sm_70` còn trong arch list không. Không còn → thử ghim `llmcompressor` bản cũ hơn cho tới khi torch giữ 2.8; ghi từng lần thử vào findings |
| `ImportError Qwen3_5ForConditionalGeneration` | Thêm `transformers>=<bản đầu có Qwen3.5>` (tra `pip index versions transformers` + changelog), build lại |
| Không có tổ hợp nào vừa có `sm_70` vừa có Qwen3.5 | Ghi findings Q6 = "không"; giữ tổ hợp mới nhất chạy được Qwen3.5 (bỏ SM < 8.0 khỏi bản đầu, theo spec §10.6) |

- [ ] **Step 4: Ghim phiên bản vào Dockerfile**

Thay dòng `RUN pip install` bằng các bản cụ thể đã chạy, ví dụ dạng:
```dockerfile
RUN pip install --no-cache-dir llmcompressor==X.Y.Z compressed-tensors==A.B.C transformers==M.N.P datasets==D.E.F
```
(X.Y.Z... là số thật in ở Step 2.) Build lại, chạy lại lệnh Step 2, kết quả phải giống.

- [ ] **Step 5: Tìm helper lưu MTP**

```bash
ssh rtx4070 'docker run --rm qtool-compress:spike bash -c "grep -rn \"def save_mtp_tensors_to_checkpoint\" /opt/conda/lib/python3*/site-packages/ || echo NOT_FOUND"'
```
Ghi vào findings Q2 (đường dẫn import hoặc NOT_FOUND).

- [ ] **Step 6: Commit**

```bash
git add images/compress/Dockerfile && git commit -m "build: pin compress image versions from spike"
```

---

### Task 3: Script quantize spike + chạy INT4-W4A16 trên rtx4070 (BF16)

Trả lời Q1 (phần quantize), Q3 (VRAM BF16), thu fixture log llmcompressor GPTQ.

**Files:**
- Create (rtx4070, không commit): `~/qtool-spike/quantize_spike.py`
- Create (repo): `tests/fixtures/llmcompressor/w4a16-gptq.log`

- [ ] **Step 1: Viết script spike**

```python
# ABOUTME: Spike script (throwaway): quantize Qwen3.5 with llmcompressor oneshot like the vLLM docs example,
# ABOUTME: printing config, calib token stats and peak memory so real logs can be captured as fixtures.
import argparse
import json
import statistics
import time

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

IGNORE = ["re:.*lm_head", "re:.*embed_tokens$", "re:.*visual.*", "re:.*model.visual.*", "re:.*linear_attn.*"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--scheme", choices=["W4A16", "W8A8"], required=True)
ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
ap.add_argument("--samples", type=int, default=512)
ap.add_argument("--seq-len", type=int, default=2048)
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = AutoConfig.from_pretrained(a.model)
print("SPIKE config", json.dumps({"arch": cfg.architectures, "dtype": str(getattr(cfg, "torch_dtype", None)),
                                  "quantized": hasattr(cfg, "quantization_config")}), flush=True)

t0 = time.time()
model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=getattr(torch, a.dtype))
tok = AutoTokenizer.from_pretrained(a.model)
print("SPIKE load_seconds", round(time.time() - t0, 1), "params", model.num_parameters(), flush=True)

t0 = time.time()
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=42).select(range(a.samples))
ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})
ds = ds.map(lambda s: tok(s["text"], padding=False, max_length=a.seq_len, truncation=True, add_special_tokens=False),
            remove_columns=ds.column_names)
lens = [len(x) for x in ds["input_ids"]]
print("SPIKE calib", json.dumps({"n": len(lens), "min": min(lens), "median": statistics.median(lens), "max": max(lens),
                                 "truncated": sum(1 for n in lens if n >= a.seq_len), "seconds": round(time.time() - t0, 1)}),
      flush=True)

if a.scheme == "W4A16":
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE, sequential_targets=["Qwen3_5DecoderLayer"])
else:
    recipe = [SmoothQuantModifier(smoothing_strength=0.8),
              GPTQModifier(targets="Linear", scheme="W8A8", ignore=IGNORE, sequential_targets=["Qwen3_5DecoderLayer"])]

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=a.seq_len, num_calibration_samples=a.samples)
print("SPIKE oneshot_seconds", round(time.time() - t0, 1), "peak_gpu_bytes", torch.cuda.max_memory_allocated(), flush=True)

model.save_pretrained(a.out, save_compressed=True)
AutoProcessor.from_pretrained(a.model).save_pretrained(a.out)
print("SPIKE saved", a.out, flush=True)
```

Ghi chú: ignore list và `sequential_targets` lấy từ recipe RedHat; calib theo docs vLLM. Script **chưa** lưu MTP (xem Task 3 Step 5).

- [ ] **Step 2: Chạy, kèm đo VRAM bằng nvidia-smi song song**

```bash
rsync -a <scratchpad>/quantize_spike.py rtx4070:qtool-spike/   # file Step 1 viết ở scratchpad của session, không trong repo
ssh rtx4070 'cd ~/qtool-spike && \
  { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > logs/w4a16-bf16.vram.csv & P=$!; } && \
  docker run --rm --gpus all --name qtool-spike -e HF_TOKEN \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/qtool-spike:/work -w /work \
    qtool-compress:spike python quantize_spike.py --scheme W4A16 --out /work/ckpt/INT4-W4A16 \
    > logs/w4a16-bf16.log 2>&1; echo EXIT $?; kill $P'
```
Expected: `EXIT 0`. Nếu lỗi: ghi dòng lỗi gốc vào findings Q1 và thử sửa đúng một chỗ (ví dụ đổi `sequential_targets`); mỗi lần thử ghi lại.

- [ ] **Step 3: Đọc số liệu**

```bash
ssh rtx4070 'cd ~/qtool-spike && grep "^SPIKE" logs/w4a16-bf16.log; sort -n logs/w4a16-bf16.vram.csv | tail -1; du -sh ckpt/INT4-W4A16; \
  du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B'
```
Ghi vào findings Q3: `peak_gpu_bytes` (torch), VRAM đỉnh (nvidia-smi), thời gian load / calib / oneshot, dung lượng trước/sau.

- [ ] **Step 4: Đối chiếu checkpoint với RedHat**

```bash
ssh rtx4070 'cd ~/qtool-spike && python3 -c "
import json,sys
a=json.load(open(\"ckpt/INT4-W4A16/config.json\"))[\"quantization_config\"]
S=sys.argv[1]; b=json.load(open(S+\"config.json\"))[\"quantization_config\"]
for k in (\"format\",\"quant_method\"): print(k, a.get(k), b.get(k))
print(\"groups\", json.dumps(a[\"config_groups\"])[:400]); print(\"rh\", json.dumps(b[\"config_groups\"])[:400])
print(\"ignore n\", len(a[\"ignore\"]), len(b[\"ignore\"]))" $(ls -d ~/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-4B-quantized.w4a16/snapshots/*/)'
```
Expected: cùng `format` (`pack-quantized`), cùng num_bits 4 / group 128 / symmetric. Khác biệt → ghi findings Q2.

- [ ] **Step 5: MTP**

Nếu Task 2 Step 5 tìm thấy `save_mtp_tensors_to_checkpoint`: thêm vào cuối script
`from <module tìm được> import save_mtp_tensors_to_checkpoint` và
`save_mtp_tensors_to_checkpoint(source_model=a.model, dest_dir=a.out)`, chạy lại Step 2, kiểm `ls ckpt/INT4-W4A16/model_mtp.safetensors`.
Nếu NOT_FOUND: ghi findings Q2 "MTP không được lưu; checkpoint không dùng được speculative MTP" — plan 2 quyết định.

- [ ] **Step 6: Copy fixture về repo và commit**

```bash
mkdir -p tests/fixtures/llmcompressor
rsync rtx4070:qtool-spike/logs/w4a16-bf16.log tests/fixtures/llmcompressor/w4a16-gptq.log
git add tests/fixtures/llmcompressor && git commit -m "test: add real llmcompressor GPTQ W4A16 log fixture"
```

---

### Task 4: Image eval + chọn chế độ eval + fixture vLLM / lm_eval

Trả lời Q4 (`max_model_len`, `gpu_memory_utilization`, `add_bos_token`, chế độ prompt), Q1 (checkpoint load được bằng vLLM 0.28), Q5 (`int8`), và tên tham số FP8 online.

**Files:**
- Create (repo): `images/eval/Dockerfile`
- Create (repo): `tests/fixtures/vllm/*.log`, `tests/fixtures/lm_eval/*`

- [ ] **Step 1: Dockerfile eval**

```dockerfile
# ABOUTME: Image cho step 4 cua qtool: vLLM 0.28 (giong bake-off) + lm-eval.
# ABOUTME: Phien ban lm-eval ghim theo spike 2026-10-30.
FROM vllm/vllm-openai:v0.28.0-cu129-ubuntu2404
RUN pip install --no-cache-dir "lm-eval[api]>=0.4.12"
ENTRYPOINT []
```

- [ ] **Step 2: Build, kiểm torch / vLLM không bị pip thay đổi**

```bash
rsync -a images/ rtx4070:qtool-spike/images/
ssh rtx4070 'cd ~/qtool-spike && docker build -t qtool-eval:spike images/eval 2>&1 | tail -3 && \
  for i in vllm/vllm-openai:v0.28.0-cu129-ubuntu2404 qtool-eval:spike; do \
    docker run --rm --entrypoint python3 $i -c "import torch,vllm;print(torch.__version__,vllm.__version__)"; done; \
  docker run --rm qtool-eval:spike python3 -c "import lm_eval;print(lm_eval.__version__)"'
```
Expected: 2 dòng torch/vLLM giống hệt nhau. Khác → ghim lm-eval bản không kéo torch, build lại. Ghim bản lm-eval vào Dockerfile.

- [ ] **Step 3: Tra tham số của backend vllm trong lm_eval và tên FP8 online**

```bash
ssh rtx4070 'docker run --rm qtool-eval:spike bash -c "
python3 -c \"import inspect,lm_eval.models.vllm_causallms as m; print(inspect.signature(m.VLLM.__init__))\";
python3 -c \"from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS as Q; print([q for q in Q if \\\"fp8\\\" in q])\""'
```
Ghi vào findings Q4: có `enable_thinking` / `chat_template_args` / `think_end_token` không; danh sách tên quantization chứa `fp8`.

- [ ] **Step 4: Chạy BF16 baseline 2 chế độ prompt, limit 50**

Biến dùng chung:
```bash
EVAL='docker run --rm --gpus all --name qtool-spike-eval -e HF_TOKEN -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/qtool-spike:/work -w /work qtool-eval:spike'
ARGS='max_model_len=4096,gpu_memory_utilization=0.85,add_bos_token=True'
```
(a) Theo docs vLLM (completion, 5-shot):
```bash
ssh rtx4070 "$EVAL lm_eval --model vllm --model_args pretrained=Qwen/Qwen3.5-4B,$ARGS \
  --tasks gsm8k --num_fewshot 5 --limit 50 --batch_size auto --output_path out/bf16-docs --log_samples \
  > ~/qtool-spike/logs/eval-bf16-docs.log 2>&1; echo EXIT \$?"
```
(b) Chat template, thinking tắt (chỉ khi Step 3 thấy tham số tương ứng; tên tham số lấy đúng từ Step 3):
```bash
ssh rtx4070 "$EVAL lm_eval --model vllm --model_args pretrained=Qwen/Qwen3.5-4B,$ARGS,<tham số tắt thinking từ Step 3> \
  --tasks gsm8k --num_fewshot 5 --limit 50 --batch_size auto --apply_chat_template --fewshot_as_multiturn \
  --output_path out/bf16-chat --log_samples > ~/qtool-spike/logs/eval-bf16-chat.log 2>&1; echo EXIT \$?"
```
Ghi findings Q4 cho từng chế độ: `exact_match` strict / flexible, thời gian, số token sinh trung bình (từ file samples), có câu nào bị cắt ở `max_model_len` không. Thử lại (a) với `add_bos_token=False`: điểm khác → ghi lại; tokenizer Qwen có `bos_token` không (`tokenizer_config.json`).

- [ ] **Step 5: Quyết định chế độ eval**

Tiêu chí theo thứ tự: (1) điểm BF16 hợp lý (không thấp bất thường do định dạng prompt), (2) thời gian full 1319 câu ước từ limit 50 ≤ 15 phút cho 4B, (3) giống docs vLLM nếu ngang nhau. Ghi lựa chọn + số liệu vào findings Q4. Không chế độ nào đạt (1) → ghi lại, hỏi Nhân trước khi đi tiếp.

- [ ] **Step 6: Eval checkpoint INT4-W4A16 tự làm: không env, `fp8`, `int8`; và RedHat với `fp8`**

Với chế độ đã chọn ở Step 5 (thay `<MODE_FLAGS>` bằng cờ của chế độ đó, ví dụ `--num_fewshot 5` cho (a)):
```bash
for v in none fp8 int8; do
  ENV=""; [ $v != none ] && ENV="-e VLLM_MARLIN_INPUT_DTYPE=$v"
  ssh rtx4070 "docker run --rm --gpus all $ENV -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/qtool-spike:/work -w /work qtool-eval:spike \
    lm_eval --model vllm --model_args pretrained=/work/ckpt/INT4-W4A16,$ARGS --tasks gsm8k <MODE_FLAGS> --limit 50 \
    --batch_size auto --output_path out/w4a16-$v --log_samples > ~/qtool-spike/logs/eval-w4a16-$v.log 2>&1; echo $v EXIT \$?"
done
ssh rtx4070 "docker run --rm --gpus all -e VLLM_MARLIN_INPUT_DTYPE=fp8 -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/qtool-spike:/work -w /work qtool-eval:spike \
  lm_eval --model vllm --model_args pretrained=RedHatAI/Qwen3.5-4B-quantized.w4a16,$ARGS --tasks gsm8k <MODE_FLAGS> --limit 50 \
  --batch_size auto --output_path out/rh-fp8 --log_samples > ~/qtool-spike/logs/eval-rh-fp8.log 2>&1; echo rh EXIT \$?"
ssh rtx4070 'cd ~/qtool-spike/logs && grep -HiE "Marlin|CompressedTensors|kernel|quantiz|Model loading took|KV cache|GPU KV" eval-*.log | cut -c1-200'
```
Ghi findings: Q1 (checkpoint tự làm load được bằng vLLM 0.28: có/không + lỗi gốc), Q5 (`int8` chạy được không, log kernel), dòng log nào chứng minh Marlin + kiểu activation. So `eval-w4a16-fp8.log` với log bake-off 26/10 (`qwen-bench/logs/run_confirm_w4a8_1026.log`, dòng `Using MarlinLinearKernel for CompressedTensorsWNA16`).

- [ ] **Step 7: FP8 online**

```bash
ssh rtx4070 "$EVAL lm_eval --model vllm --model_args pretrained=Qwen/Qwen3.5-4B,$ARGS,quantization=<tên fp8 từ Step 3> \
  --tasks gsm8k <MODE_FLAGS> --limit 50 --batch_size auto --output_path out/fp8-online --log_samples \
  > ~/qtool-spike/logs/eval-fp8-online.log 2>&1; echo EXIT \$?"
```

- [ ] **Step 8: Copy fixture, ghim lm-eval, commit**

```bash
mkdir -p tests/fixtures/vllm tests/fixtures/lm_eval
rsync rtx4070:'qtool-spike/logs/eval-*.log' tests/fixtures/lm_eval/
rsync -a rtx4070:qtool-spike/out/ tests/fixtures/lm_eval/out/
git add images/eval/Dockerfile tests/fixtures && git commit -m "test: add real vLLM and lm_eval fixtures; pin eval image"
```
(Log lm_eval chứa cả log vLLM vì chạy cùng process; plan 2 tách fixture theo parser.)

---

### Task 5: INT8-W8A8 (SmoothQuant + GPTQ)

Trả lời Q1 cho recipe thứ 2; thu fixture log SmoothQuant.

**Files:**
- Create (repo): `tests/fixtures/llmcompressor/w8a8-smoothquant-gptq.log`

- [ ] **Step 1: Chạy quantize W8A8**

```bash
ssh rtx4070 'cd ~/qtool-spike && \
  { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > logs/w8a8-bf16.vram.csv & P=$!; } && \
  docker run --rm --gpus all --name qtool-spike -e HF_TOKEN \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/qtool-spike:/work -w /work \
    qtool-compress:spike python quantize_spike.py --scheme W8A8 --out /work/ckpt/INT8-W8A8 \
    > logs/w8a8-bf16.log 2>&1; echo EXIT $?; kill $P'
```
Rủi ro đã biết: mapping mặc định của SmoothQuant có thể không khớp lớp GDN (`linear_attn`) của Qwen3.5. Lỗi → ghi dòng lỗi gốc; thử `SmoothQuantModifier(smoothing_strength=0.8, mappings=[[["re:.*self_attn.q_proj", "re:.*self_attn.k_proj", "re:.*self_attn.v_proj"], "re:.*input_layernorm"], [["re:.*mlp.gate_proj", "re:.*mlp.up_proj"], "re:.*post_attention_layernorm"]])` (chỉ áp cho layer full attention + MLP); ghi kết quả từng lần thử vào findings Q1.

- [ ] **Step 2: Eval INT8-W8A8 (chế độ Task 4 Step 5, limit 50) + ghi số liệu VRAM như Task 3 Step 3**

```bash
ssh rtx4070 "docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/qtool-spike:/work -w /work qtool-eval:spike \
  lm_eval --model vllm --model_args pretrained=/work/ckpt/INT8-W8A8,$ARGS --tasks gsm8k <MODE_FLAGS> --limit 50 \
  --batch_size auto --output_path out/w8a8 --log_samples > ~/qtool-spike/logs/eval-w8a8.log 2>&1; echo EXIT \$?"
ssh rtx4070 'cd ~/qtool-spike && grep "^SPIKE" logs/w8a8-bf16.log; sort -n logs/w8a8-bf16.vram.csv | tail -1; du -sh ckpt/INT8-W8A8'
```

- [ ] **Step 3: Copy fixture và commit**

```bash
rsync rtx4070:qtool-spike/logs/w8a8-bf16.log tests/fixtures/llmcompressor/w8a8-smoothquant-gptq.log
rsync rtx4070:qtool-spike/logs/eval-w8a8.log tests/fixtures/lm_eval/
git add tests/fixtures && git commit -m "test: add real SmoothQuant+GPTQ W8A8 fixtures"
```

---

### Task 6: Hệ số VRAM — FP32 trên rtx4070, và 9B

Trả lời Q3 (công thức ngưỡng preflight).

- [ ] **Step 1: W4A16 FP32 với 4B trên rtx4070**

Như Task 3 Step 2 nhưng `--dtype float32 --out /work/ckpt/INT4-W4A16-fp32`, log `logs/w4a16-fp32.log`, VRAM `logs/w4a16-fp32.vram.csv`. OOM → ghi lại (cũng là một số liệu).

- [ ] **Step 2: W4A16 BF16 với 9B (chỉ khi còn ≥ 40 GB đĩa trống)**

Tải `Qwen/Qwen3.5-9B` như Task 1 Step 2; chạy như Task 3 Step 2 với `--model Qwen/Qwen3.5-9B --out /work/ckpt/9B-INT4-W4A16`, log `logs/9b-w4a16-bf16.log`. Theo dõi RAM host: `free -g` trước / sau (host có 30 GB RAM).

- [ ] **Step 3: Tính hệ số**

Với mỗi lần chạy: `hệ số = VRAM đỉnh (nvidia-smi) / dung lượng safetensors model gốc`. Ghi bảng (model, dtype, VRAM đỉnh, RAM host đỉnh, hệ số) vào findings Q3, kèm công thức đề xuất cho preflight.

---

### Task 7: SM 7.0 (V100 Apollo) và SM 7.5

Trả lời Q6, Q7. **Hỏi Nhân GPU Apollo nào được dùng trước Step 2.**

- [ ] **Step 1: Kiểm vLLM 0.28 có build SM 7.5 / 7.0**

```bash
ssh rtx4070 'docker run --rm --entrypoint bash vllm/vllm-openai:v0.28.0-cu129-ubuntu2404 -c "
f=\$(python3 -c \"import vllm,os;print(os.path.dirname(vllm.__file__))\")/_C.abi3.so; cuobjdump --list-elf \$f 2>/dev/null | grep -o \"sm_[0-9]*\" | sort -u || echo NO_CUOBJDUMP"'
```
NO_CUOBJDUMP → thay bằng `python3 -c "import torch;print(torch.cuda.get_arch_list())"` trong image và ghi rõ đây là arch list của torch, không phải của `vllm._C`. Ghi findings Q7.

- [ ] **Step 2: Hỏi Nhân GPU Apollo; chuẩn bị image trên Apollo**

Sau khi có GPU index `G`:
```bash
ssh rtx4070 'docker save qtool-compress:spike -o ~/qtool-spike/compress.tar'
ssh apollo 'mkdir -p /u01/nhandlt2/qtool-spike'
scp -3 rtx4070:qtool-spike/compress.tar apollo:/u01/nhandlt2/qtool-spike/
ssh apollo 'docker load -i /u01/nhandlt2/qtool-spike/compress.tar'
```
(Image load trên Apollo ghi vào docker data-root ở đĩa gốc chậm → có thể mất 25+ phút; chạy nền, theo dõi.)

- [ ] **Step 3: GPTQ FP32 4B trên V100**

```bash
ssh apollo 'mkdir -p /u01/nhandlt2/qtool-spike/{ckpt,logs} && cd /u01/nhandlt2/qtool-spike && \
  docker run --rm --gpus device=G --name qtool-spike -v /u01/nhandlt2/huggingface:/root/.cache/huggingface \
  -v /u01/nhandlt2/qtool-spike:/work -w /work qtool-compress:spike \
  python quantize_spike.py --scheme W4A16 --dtype float32 --out /work/ckpt/INT4-W4A16-v100 > logs/w4a16-v100.log 2>&1; echo EXIT $?'
```
(Copy `quantize_spike.py` sang `/u01/nhandlt2/qtool-spike/` trước. Model `Qwen/Qwen3.5-4B` tải vào `/u01/nhandlt2/huggingface`.)

- [ ] **Step 4: Eval checkpoint V100 trên rtx4070**

rsync `ckpt/INT4-W4A16-v100` về `rtx4070:qtool-spike/ckpt/`, eval như Task 4 Step 6 (không env). Điểm phải gần điểm checkpoint rtx4070 (cùng limit 50). Ghi findings Q6: chạy được / không, thời gian, VRAM, điểm.

- [ ] **Step 5: Fixture nvidia-smi nhiều GPU**

```bash
ssh apollo 'nvidia-smi --query-gpu=index,name,compute_cap,memory.used,memory.total,memory.free --format=csv,noheader,nounits' > tests/fixtures/nvidia-smi/apollo-8gpu.csv
ssh apollo 'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits' > tests/fixtures/nvidia-smi/apollo-8gpu-apps.csv
ssh apollo 'nvidia-smi --query-gpu=index,uuid --format=csv,noheader' > tests/fixtures/nvidia-smi/apollo-8gpu-uuid.csv
rsync apollo:/u01/nhandlt2/qtool-spike/logs/w4a16-v100.log tests/fixtures/llmcompressor/w4a16-gptq-v100-fp32.log
git add tests/fixtures && git commit -m "test: add Apollo multi-GPU and V100 FP32 fixtures"
```

---

### Task 8: Findings và dọn dẹp

**Files:**
- Create (repo): `docs/spike/2026-10-30-findings.md`

- [ ] **Step 1: Viết findings** — mỗi câu hỏi một mục, mỗi số liệu ghi nguồn (file log / lệnh):

```markdown
# qtool Spike Findings — 2026-10-30

## Q1. Phiên bản: llmcompressor hiểu Qwen3.5 và checkpoint load được bằng vLLM 0.28
## Q2. Ignore list, MTP, so với checkpoint RedHat
## Q3. VRAM / RAM đỉnh, thời gian, hệ số → công thức ngưỡng preflight
## Q4. Eval: chế độ prompt, max_model_len, gpu_memory_utilization, add_bos_token, thời gian full gsm8k ước tính, tên FP8 online
## Q5. VLLM_MARLIN_INPUT_DTYPE=int8 trên vLLM 0.28
## Q6. Quantize trên SM 7.0 (V100, FP32)
## Q7. vLLM 0.28: SM 7.5 / 7.0
## Q8. Fixture đã thu (đường dẫn trong tests/fixtures/, dòng log nào parser cần bắt)
## Ảnh hưởng tới spec (mục nào cần sửa)
```

- [ ] **Step 2: Dọn dẹp** — xoá checkpoint spike lớn sau khi Nhân xem findings (hỏi trước): `~/qtool-spike/ckpt/*` trên rtx4070, `/u01/nhandlt2/qtool-spike/` trên Apollo. Không xoá model trong HF cache.

- [ ] **Step 3: Commit**

```bash
git add docs/spike && git commit -m "docs: add spike findings"
```
