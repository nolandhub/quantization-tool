# qtool — Design Spec

Ngày: 2026-10-30 · Trạng thái: bản 3.2 (spike 1 + 2, `docs/spike/2026-10-30-findings.md`; 3.1 = chi tiết kiểm chứng lúc viết plan 2; 3.2 = NVFP4 + scheme chia nhóm theo checkpoint, mục 11), Nhân đã duyệt v3 · Repo: `~/ws-DoNhan/qtool` (local) · máy phát triển và kiểm chứng: `rtx4070:~/qtool` (RTX 4070 Ti SUPER, SM 8.9)

## 1. Mục tiêu

CLI đơn giản gói 4 bước quantize theo flow LLM Compressor của vLLM
([https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/)):

1. Loading the model
2. Preparing calibration data
3. Applying quantization
4. Evaluating accuracy in vLLM

Dev chỉ cần chỉ tên model và scheme. Console log bằng tiếng Anh, từng bước in rõ đang làm gì,
tiến trình tới đâu, mất bao lâu. **Parse, không tự chế:** mọi tiến trình và số liệu hiển thị lấy
từ output thật của llmcompressor / vLLM / lm_eval hoặc từ file (config.json, results.json);
tool chỉ tự đo thời gian và tự tính Δ pp, recovery %, ETA (ghi rõ `est.`) từ các số đã parse.

Lý do phải tự làm: vLLM/LLM Compressor chỉ cung cấp từng mảnh — `llmcompressor.oneshot()`
(Python API, không có CLI quantize; console script duy nhất là `llmcompressor.trace` và
`reindex_fused_weights` đã deprecated), `lm_eval` (CLI riêng, không phải dependency của
llmcompressor), cờ runtime `VLLM_MARLIN_INPUT_DTYPE` của vLLM. qtool là lớp keo nối các mảnh.

## 2. Phạm vi

### Trong phạm vi

- 2 lệnh con: `qtool quantize`, `qtool eval`.
- 6 scheme (bảng mục 3). Mỗi scheme là 1 dòng kết quả; qtool tự suy ra checkpoint cần build, chỉ eval đúng scheme được yêu cầu.
- Mọi model transformers 5.14 load được (không `trust_remote_code`) và vLLM 0.28 chạy được: recipe **suy từ
  `config.json` + cấu trúc module** (mục 3 "Recipe chung"), không tra bảng theo tên. Đã kiểm chứng (spike Q9):
  Qwen3.5 (hybrid GDN + vision + MTP), Phi3 (dense), OLMoE (MoE 64 expert), SmolVLM (đa phương thức, text Llama).
  Kiến trúc khác chạy với nhãn `[unverified architecture]`.
- Phần cứng: GPU NVIDIA bất kỳ có Docker + NVIDIA Container Toolkit; 1 GPU mỗi lượt chạy, chọn bằng
  `--gpu` (bắt buộc khi máy có ≥ 2 GPU). Scheme nào eval được trên GPU nào: bảng mục 3.
- Eval bằng `lm_eval --model vllm`, task bất kỳ của lm_eval, mặc định `gsm8k`.

### Ngoài phạm vi (non-goals)

- Không dừng / bật container hay service nào. Dev tự làm; tool chỉ kiểm tra và báo.
- Không tích hợp qbench, voice, rag hay bất cứ gì của `qwen-bench/`.
- Không MXFP4, NVFP4A16 (NVFP4 chỉ weight), W8A16, INT4 W4A4, INT4 bất đối xứng / AWQ, recipe mixed-precision, checkpoint FP8 lưu sẵn.
  INT4 W4A4: vLLM không có kernel, Hopper/Blackwell bỏ tensor core INT4. INT4 bất đối xứng: Marlin W4A8 không nhận.
- Không multi-GPU / tensor parallel.
- Không eval trên SM 7.0 (V100): vLLM 0.28 bỏ SM 7.0. vLLM 0.18 (bản cuối có SM 7.0) **chạy được** Qwen3.5
  (kiểm 02/11 trên apollo: 4B và 9B fp16), nhưng trên SM 7.0 chỉ nhận INT4 dạng `gptq` (compressed-tensors WNA16 cần
  SM ≥ 7.5) → checkpoint qtool không chạy được trên V100, và qtool không hỗ trợ image vLLM 0.18. V100 chỉ quantize
  (`--no-eval`), eval ở máy khác.
- Không kiểm định thống kê (McNemar, p-value). Theo cách làm gốc: bảng lm_eval có stderr + recovery %.
- Không cài `flash-linear-attention` (transformers log `fast path is not available` → calib lớp GDN chạy torch thuần;
  có thể nhanh hơn nhưng chưa thử, ghi vào hướng mở rộng).

### Hướng mở rộng (ghi lại, chưa làm)

- Preset mixed-precision theo nhóm layer (giữ attention/GDN 8 bit, MLP 4 bit), kiểu `RedHatAI/GLM-5.3-MXFP4xFP8`.
- Tập calib tiếng Việt hoặc trộn (`--calib-mix`).
- Recipe `FP8_DYNAMIC` lưu checkpoint.
- Multi-GPU (tensor parallel) cho model không vừa 1 GPU.
- Kiểm chứng thêm kiến trúc gated (Llama 3.x, Gemma 3) khi có `HF_TOKEN`.
- Model cần `trust_remote_code`.
- `flash-linear-attention` trong image compress để rút thời gian calib Qwen3.5.
- Xuất định dạng `gptq` (qweight / qzeros / scales / g_idx) từ `INT4-W4A16` để chạy trên V100 với vLLM 0.18.
  02/11: chưa cần — `Intel/Qwen3.5-9B-int4-AutoRound` (dạng GPTQ, quantize cả GDN) chạy trên V100: gsm8k −1.4 pp so với
  fp16 (không có ý nghĩa thống kê), decode ×2.7 khi ít request, kernel `gptq_gemm` cho kết quả như Marlin.

## 3. Scheme

Mỗi scheme là 1 dòng trong bảng kết quả; checkpoint là việc của qtool. Scheme chia 3 nhóm theo checkpoint
(`--help`, README, thông báo lỗi và kế hoạch đều trình bày theo nhóm này):

| `--scheme`       | Nhóm                     | Weight        | Activation | Checkpoint                                                                                                               | Eval runtime                             |
| ------------------ | ------------------------- | ------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| `int4-w4a16`     | tạo checkpoint           | INT4 sym g128 | BF16       | build `INT4-W4A16`: `GPTQModifier(targets="Linear", scheme="W4A16", dampening_frac=0.05)`                            | mặc định                              |
| `int8-w8a8`      | tạo checkpoint           | INT8          | INT8       | build `INT8-W8A8`: `SmoothQuantModifier(smoothing_strength=0.8)` + `GPTQModifier(targets="Linear", scheme="W8A8")` | mặc định                              |
| `nvfp4-w4a4`     | tạo checkpoint           | FP4 E2M1 g16  | FP4 E2M1   | build `NVFP4`: `QuantizationModifier(targets="Linear", scheme="NVFP4")` (scale nhóm FP8 E4M3 + global scale FP32)   | mặc định                              |
| `int4-w4a8-fp8`  | dùng lại `INT4-W4A16` | INT4          | FP8        | dùng chung `INT4-W4A16` (quantize 1 lần)                                                                             | `VLLM_MARLIN_INPUT_DTYPE=fp8`          |
| `int4-w4a8-int8` | dùng lại `INT4-W4A16` | INT4          | INT8       | dùng chung `INT4-W4A16` (quantize 1 lần)                                                                             | `VLLM_MARLIN_INPUT_DTYPE=int8`         |
| `fp8-w8a8`       | không checkpoint         | FP8           | FP8        | không có: vLLM quantize model BF16 lúc load                                                                           | model BF16 gốc + `--quantization fp8` |

Ghi chú:

- Không scheme nào quantize riêng cho W4A8: `--scheme int4-w4a8-fp8` build `INT4-W4A16` (hoặc dùng lại nếu có) rồi chỉ eval W4A8;
  `--scheme int4-w4a16,int4-w4a8-fp8` quantize 1 lần, eval 2 lượt. Người chỉ cần W4A8 không phải chờ thêm lượt eval W4A16.
- W4A8 trên Ada = checkpoint INT4 đối xứng g128 + Marlin lượng tử hoá activation động lúc chạy
  (đã kiểm chứng trong bake-off 26/10, `qwen-bench/docs/handoff-2026-10-27.md`). Checkpoint gắn
  nhãn `CompressedTensorsW4A8Fp8` đòi SM 9.0 nên không dùng. INT4 bất đối xứng (zero-point, AWQ) không chạy được W4A8.
- `fp8-w8a8` vẫn là quantize thật, chỉ là vLLM làm lúc load (scale động, không cần calib); qtool không chạy step 1–3 cho nó.
  Giữ trong `quantize` để FP8 nằm cùng bảng so sánh với các scheme khác.
- Quantize luôn đi từ model gốc chưa quantize.
- `int4-w4a8-int8` đã chạy thật trên SM 8.9 (spike Q5: gsm8k limit 50 → 0.80).
- NVFP4 và INT4 là hai kiểu số khác nhau (lưới INT4 đều −8…7, scale BF16 / 128 weight ≈ 4.13 bit/weight;
  lưới FP4 E2M1 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, scale FP8 / 16 weight + global FP32 ≈ 4.5 bit/weight) → checkpoint riêng,
  không dùng chung với `INT4-W4A16`.
- NVFP4 không dùng GPTQ mặc định: nhóm 16 + lưới không đều làm sai số làm tròn thẳng nhỏ hơn nhiều so với INT4 g128,
  và với W4A4 sai số chủ yếu nằm ở activation mà GPTQ không xử lý. `QuantizationModifier` vẫn cần tập calib (observer
  `static_minmax` đo global scale activation) → dùng chung cache calib. Acceptance 02/11: RTN giữ 99.9 % (−0.1 pp) → không cần GPTQ, mặc định giữ `QuantizationModifier`.
- Đã kiểm trên image hiện tại (02/11): llmcompressor 0.13.0 / compressed-tensors 0.18.0 có preset `NVFP4`
  (`group_size=16`, `strategy=tensor_group`, `scale_dtype=float8_e4m3fn`); vLLM 0.28 map NVFP4 sang `CompressedTensorsW4A4Fp4`
  (min capability 75). Không đổi image.

### Recipe chung (spike Q9)

`quantize.py` suy recipe cho mọi model theo thứ tự sau, in kết quả ở `[1/4]` / `[3/4]` để dev thấy:

| Bước          | Quy tắc                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Load            | có `vision_config` → `AutoModelForImageTextToText`, còn lại `AutoModelForCausalLM`; dtype gốc đọc `dtype` / `torch_dtype` ở cấp trên cùng rồi `text_config`                                                                                                                                                                                                                    |
| MoE             | `linearize_moe(model)` của llmcompressor **trước khi** suy ignore (transformers v5 gộp expert thành tensor 3D, không phải `Linear`)                                                                                                                                                                                                                                                   |
| Ignore          | luôn `re:.*lm_head`; module vision/projector (tên bắt đầu `visual`, `vision_tower`, `vision_model`, `vision_encoder`, `multi_modal_projector`, `mm_projector`, `connector`); MoE: `Linear` có `out_features` ∈ {số expert, 1} (router); hybrid (`layer_types` có loại khác attention): `linear_attn` / `mamba` / `mixer` / `ssm`; cộng `--ignore` của dev |
| Layer tuần tự | `ModuleList` lớn nhất ngoài vision có class nằm trong `_no_split_modules`; truyền qua `oneshot(sequential_targets=[...])`, **không** vào `GPTQModifier`                                                                                                                                                                                                                          |
| SmoothQuant     | mapping tự suy của llmcompressor                                                                                                                                                                                                                                                                                                                                                                     |
| Calib           | render chat template, **kiểm nội dung tin nhắn còn trong text**; mất → thử `content=[{"type":"text",...}]`; vẫn mất / không có template → text thô. In dạng đã dùng + thống kê token                                                                                                                                                                                        |
| `oneshot`     | truyền tokenizer làm `processor=` (không khởi tạo processor đa phương thức — kéo gói phụ như `num2words`)                                                                                                                                                                                                                                                                            |
| Lưu            | `save_pretrained(save_compressed=True)` + tokenizer + copy `*.json/*.jinja/*.txt/*.model` còn thiếu từ snapshot gốc; `text_config` có khoá chứa `mtp` / `nextn` với giá trị > 0 → `save_mtp_tensors_to_checkpoint`                                                                                                                                                              |

Kiến trúc đã kiểm chứng (recipe suy ra → checkpoint load được vLLM + Marlin):

| Kiến trúc                         | Model thử                | Layer tuần tự         | Ignore suy ra                            | Linear quantize                |
| ----------------------------------- | ------------------------- | ----------------------- | ---------------------------------------- | ------------------------------ |
| `Qwen3_5ForConditionalGeneration` | Qwen3.5-4B, 9B            | `Qwen3_5DecoderLayer` | lm_head, visual, linear_attn             | 128 (trùng checkpoint RedHat) |
| `Phi3ForCausalLM`                 | Phi-4-mini-instruct       | `Phi3DecoderLayer`    | lm_head                                  | 128                            |
| `OlmoeForCausalLM`                | OLMoE-1B-7B-0125-Instruct | `OlmoeDecoderLayer`   | lm_head (router không phải `Linear`) | 3,136                          |
| `SmolVLMForConditionalGeneration` | SmolVLM2-2.2B-Instruct    | `LlamaDecoderLayer`   | lm_head, vision_model, connector         | 168                            |

Kiến trúc khác: in `[unverified architecture]` kèm ignore list + layer tuần tự suy ra. Eval là bước xác nhận
→ kiến trúc chưa kiểm chứng mà `--no-eval` thì cảnh báo. Thiếu layer tuần tự (không có `ModuleList` phù hợp) → exit 2.

### Phần cứng

Nguyên tắc:

- **Quantize (step 1–3) chạy đúng trên mọi GPU.** Phần forward calib cần một kiểu số biểu diễn được
  model gốc không tràn: BF16 hoặc FP32. Phần toán GPTQ / SmoothQuant vốn tính FP32. Kiểu đích
  (INT4 / INT8 / FP8) chỉ là phép làm tròn, không cần phần cứng hỗ trợ. Tensor core BF16 (SM ≥ 8.0)
  chỉ giúp nhanh hơn; không có thì tính FP32 trên CUDA core (chậm hơn, tốn gấp đôi bộ nhớ).
  Không bao giờ tự chọn FP16 (model train BF16 có thể tràn FP16 → calib sai ngầm).
  Compute FP32 → trước khi lưu cast phần không quantize về dtype gốc (spike Q6: không cast thì checkpoint
  lưu FP32, 9.3 GB thay vì 5.2 GB; vLLM vẫn tự `Downcasting torch.float32 to torch.bfloat16`).
  Đã chạy thật trên V100: 4B W4A16 FP32 mất 2 giờ 40 phút (4070 BF16: 15 phút), eval trên rtx4070 0.80.
- **Eval (step 4) cần kernel vLLM cho đúng định dạng** → giới hạn theo SM (bảng dưới).
- **So với baseline** chỉ hợp lệ khi cùng GPU, cùng image eval, cùng task / limit / số shot (mục 6, quy tắc cache).

Bảng khả năng eval theo compute capability. ✓ = native, ⚠ = chạy nhưng khác bản chất (cảnh báo),
✗ = không chạy (preflight chặn). Đã kiểm chứng trên phần cứng thật: cột SM 8.9 (rtx4070) và ô quantize
SM 7.0 (V100, spike Q6). Các ô khác theo docs vLLM + code vLLM 0.28 → tool in nhãn `[per vLLM docs, unverified]`.

| Scheme \ SM          | 7.0 (V100)     | 7.5 (T4, 20xx)              | 8.0 / 8.6 (A100, 30xx)                    | **8.9 (L4, 40xx)**    | 9.0 (H100)                        | 10.0 (B200)                    | 12.0 (50xx) |
| -------------------- | -------------- | --------------------------- | ----------------------------------------- | --------------------------- | --------------------------------- | ------------------------------ | ----------- |
| `int4-w4a16`       | ✗             | ✗ (docs: INT4 cần > 8.0)  | ✓                                        | ✓                          | ✓                                | ✓                             | ✓          |
| `int4-w4a8-*`      | ✗             | ✗                          | ✗                                        | ✓                          | ✗ (Marlin W4A8 chậm hơn W4A16) | ✗                             | ✓          |
| `int8-w8a8`        | ✗             | ✓                          | ✓                                        | ✓                          | ✓                                | ✗ (docs: không INT8 ≥ 10.0) | ✗          |
| `nvfp4-w4a4`       | ✗             | ⚠ weight-only (Marlin FP4) | ⚠ weight-only (Marlin FP4)               | ⚠ weight-only (Marlin FP4) | ⚠ weight-only (Marlin FP4)       | ✓ FP4×FP4                    | ✓ FP4×FP4 |
| `fp8-w8a8`         | ✗             | ✗                          | ⚠ chạy dạng W8A16 weight-only (Marlin) | ✓                          | ✓                                | ✓                             | ✓          |
| baseline BF16        | ✗             | ⚠ vLLM ép FP16            | ✓                                        | ✓                          | ✓                                | ✓                             | ✓          |
| quantize (step 1–3) | ⚠ FP32, chậm | ⚠ FP32, chậm              | ✓ BF16                                   | ✓ BF16                     | ✓ BF16                           | ✓ BF16                        | ✓ BF16     |

Spike Q7: kernel vLLM 0.28 có ELF `sm_75 … sm_120a`, không có `sm_70` → cột SM 7.0 ✗ cho mọi eval là đúng;
cột SM 7.5 có kernel nhưng chưa chạy thật (không có máy) → giữ nhãn unverified.

Hàng `nvfp4-w4a4`: ⚠ = vLLM chạy weight FP4 qua Marlin, activation giữ BF16 (thực chất W4A16, tốc độ ≈ `int4-w4a16`,
checkpoint lớn hơn ~9 %). Ô SM 8.9 đã kiểm chứng (acceptance 02/11, `docs/spike/2026-11-02-nvfp4-acceptance.md`:
Qwen3.5-4B gsm8k full 0.804 vs BF16 0.805, `MarlinNvFp4LinearKernel`); ô SM 7.5 theo code vLLM 0.28 (`is_fp4_marlin_supported` = SM ≥ 7.5). FP4×FP4 thật (✓) chỉ có
trên Blackwell (10.x, 12.x): chưa có máy → `[per vLLM docs, unverified]`. Preflight cảnh báo, vd
`nvfp4-w4a4 on SM 8.9 runs as weight-only FP4 (Marlin), activations stay BF16`.

## 4. CLI

### `qtool quantize MODEL --scheme SCHEMES [options]`

```
positional:
  MODEL                    HF repo id hoặc thư mục local của model gốc (vd Qwen/Qwen3.5-4B)

required:
  --scheme SCHEMES         phân tách bằng dấu phẩy: int4-w4a16, int8-w8a8, nvfp4-w4a4,
                           int4-w4a8-fp8, int4-w4a8-int8, fp8-w8a8
                           (--help in theo 3 nhóm: build a checkpoint / reuse INT4-W4A16 / no checkpoint)

calibration (step 2):
  --calib SOURCE           HF dataset id[:split] hoặc file .jsonl; mỗi dòng có cột `messages` (chat) hoặc `text`
                           [default: HuggingFaceH4/ultrachat_200k:train_sft]
  --calib-samples N        số mẫu                              [default: 512]
  --calib-seq-len N        số token tối đa mỗi mẫu             [default: 2048]

recipe (step 3):
  --ignore PATTERNS        thêm pattern ignore (regex `re:...` hoặc tên module), phân tách dấu phẩy;
                           cộng vào ignore list suy ra, không thay thế

evaluation (step 4):
  --eval TASKS             task lm_eval, phân tách dấu phẩy    [default: gsm8k]
  --eval-limit N|full      số câu tối đa mỗi task              [default: full]
  --no-eval                bỏ step 4

run control:
  --output-dir DIR         thư mục runs                        [default: ~/qtool/runs]
  --on-error MODE          continue | stop                     [default: continue]
  --force                  quantize lại dù đã có checkpoint
  --dry-run                in kế hoạch rồi thoát
  --skip-preflight         bỏ kiểm tra VRAM (dev tự chịu)

hardware:
  --gpu N                  index GPU (theo nvidia-smi); bắt buộc khi máy có ≥ 2 GPU
  --compute-dtype D        auto | bfloat16 | float32           [default: auto]
                           auto: SM ≥ 8.0 → bfloat16, SM < 8.0 → float32 (không bao giờ float16)
  --hf-cache DIR           HF cache mount vào container        [default: $HF_HOME, else ~/.cache/huggingface]
```

### `qtool eval MODEL [options]`

MODEL là bất kỳ model nào vLLM đọc được: checkpoint qtool đã tạo, checkpoint quantize sẵn trên HF,
hoặc model BF16 gốc.

```
  --act fp8|int8           W4A8 qua Marlin; chỉ hợp lệ với checkpoint INT4 sym group-wise trên SM 8.9 / 12.0
  --quantization fp8       vLLM quantize FP8 online
  --baseline MODEL         eval thêm model này (có cache) để in Δ pp và recovery %
  --eval, --eval-limit, --output-dir, --dry-run, --skip-preflight, --gpu, --hf-cache   như quantize
```

`--act` và `--quantization` loại trừ nhau. `qtool quantize` gọi lại đúng code của `qtool eval` cho step 4.

Thông báo `unknown scheme '<x>'` liệt kê scheme theo 3 nhóm như `--help`.

### Exit code

| Code | Nghĩa                                       |
| ---- | -------------------------------------------- |
| 0    | mọi scheme OK                               |
| 1    | ≥ 1 scheme FAILED (`--on-error continue`) |
| 2    | lỗi tham số / preflight, chưa chạy gì   |
| 130  | Ctrl-C                                       |

## 5. Kiến trúc

CLI Python trên host (chỉ thư viện chuẩn) điều khiển 2 Docker image; mỗi bước nặng chạy trong
một container `--rm` tên `qtool-<pid>`. stdout/stderr container được đọc từng dòng: ghi nguyên
văn vào file log, parse, in dòng đã lọc ra console.

```
qtool (host)
  ├─ docker run qtool-compress:cu129 | :cu126  → step 1–3 (transformers + llmcompressor)
  └─ docker run qtool-eval                     → step 4 (vLLM 0.28 + lm-eval)
```

Lý do 2 image: docs vLLM khuyên tách môi trường llmcompressor và vLLM; dùng đúng image vLLM 0.28
của bake-off để số eval so được với số cũ.

Phiên bản ghim (spike Q1, Q6):

| Image                    | Base                                                              | Ghim                                                                                                |
| ------------------------ | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `qtool-compress:cu129` | `python:3.11-slim` + `gcc libc6-dev` (Triton cần C compiler) | torch 2.13.0 (index cu129, arch sm_75–sm_120), llmcompressor 0.13.0, transformers 5.14.1, datasets |
| `qtool-compress:cu126` | như trên                                                        | torch 2.13.0 (index cu126, arch sm_50–sm_90, **có sm_70**)                                  |
| `qtool-eval`           | `vllm/vllm-openai:v0.28.0-cu129-ubuntu2404`, `ENTRYPOINT []`  | `lm-eval[api]==0.4.13` cài với constraint `torch                                                |

- Chọn image compress theo SM: SM < 7.5 → `cu126`; còn lại → `cu129`. SM ≥ 10.0 chỉ có ở `cu129`.
- Không lên llmcompressor 0.15: kéo torch 2.14 cu130 (bỏ sm_70) và transformers ≥ 5.15. llmcompressor ≤ 0.11 chưa biết Qwen3.5.
- Checkpoint compressed-tensors 0.18.0 (image compress) load được bằng compressed-tensors 0.17.0 (image eval) — spike Q1.

Container:

- GPU: `docker run --gpus device=N` → container chỉ thấy đúng 1 GPU đã chọn (không thể bị rải sang card khác).
- Mount: `--hf-cache` và thư mục runs vào cả 2 container; `HF_TOKEN` truyền qua nếu có.
- Container chạy **root** (như các service khác; HF cache trên rtx4070 thuộc root nên `--user` không ghi được — smoke test 01/11).
  Sau **mỗi** container (kể cả lỗi / Ctrl-C) tool chạy `docker run --rm --entrypoint chown <image> -R $(id -u):$(id -g) /runs /qtool-home`
  → checkpoint, log, cache trong thư mục runs thuộc dev (spike: root + safetensors mode 600, dev không đọc được). Không đụng HF cache.
- `HOME=/qtool-home` mount từ `<output-dir>/.home` → cache Triton / torch.compile / vLLM giữ giữa các lượt và thuộc dev.
- Preflight chạy 1 container `probe.py` (image eval; `--no-eval` → image compress) để đọc metadata model và kiểm task lm_eval
  (~30 s) — host không cần transformers / mạng HF.
- Container compress chạy với `--memory` = RAM khả dụng lúc preflight − 2 GB: quantize vượt RAM thì container bị kill
  (exit 137, báo lỗi rõ), không kéo sập service khác trên host (spike: FP32 4B bị kill ở trần 24 GB).

### Cấu trúc repo

```
qtool/
├── qtool/
│   ├── cli.py          argparse, validate tham số
│   ├── schemes.py      bảng scheme: recipe, checkpoint dùng chung, env/args eval
│   ├── plan.py         lập kế hoạch, gộp checkpoint, kiểm tra cache
│   ├── preflight.py    parse nvidia-smi / docker info / df, kiểm tra điều kiện
│   ├── runner.py       docker run, stream dòng, log file, Ctrl-C, thư mục tạm
│   ├── parsers.py      regex log llmcompressor / vLLM / lm_eval, đọc results.json, dòng @qtool
│   ├── console.py      header bước, thời gian, định dạng
│   ├── report.py       bảng tổng hợp, Δ pp, recovery %, summary.json
│   ├── images.py       tag image = hash Dockerfile, chọn cu126/cu129, build khi thiếu
│   └── flow.py         điều phối quantize / eval: header bước, render sự kiện, --on-error
├── images/
│   ├── compress/Dockerfile
│   ├── eval/Dockerfile
│   └── scripts/               mount read-only vào /qtool của cả 2 container (sửa script không phải build lại image)
│       ├── recipe.py          suy recipe chung (load class, linearize MoE, ignore, layer tuần tự, MTP, calib template)
│       ├── quantize.py        container compress: load → calib → oneshot → save
│       └── probe.py           preflight: metadata model (config, số tham số, remote code, đã quantize, chat template), task lm_eval
├── bin/qtool           shim chạy `python3 -m qtool` từ repo (không cần pip install)
├── tests/
├── docs/specs/
├── sync.sh             rsync local ↔ rtx4070:~/qtool
└── README.md
```

### Nguồn thông tin hiển thị

| Loại                                                                                                                    | Cách lấy                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sự kiện của thư viện (tqdm llmcompressor, dòng GPTQ, dòng kernel vLLM, tiến trình lm_eval)                      | regex trên output gốc; regex viết từ fixture log thật của spike                                                                                                                          |
| Sự kiện script mình đọc trong container (config.json, thống kê token calib, dung lượng checkpoint, VRAM đỉnh) | `quantize.py` / `probe.py` in `@qtool {json}`; host tìm `@qtool ` ở **bất kỳ đâu** trong dòng (tqdm viết `\r` không xuống dòng nên dòng có thể dính tiền tố) |
| Kết quả eval                                                                                                           | `results.json` + file sample của lm_eval                                                                                                                                                    |
| Thời gian, Δ pp, recovery %                                                                                            | host tự đo / tự tính từ số đã parse                                                                                                                                                    |

Không tìm thấy dòng mong đợi → in `unknown (not found in <nguồn>)`, không bao giờ đoán.

Dòng log thật cần parse (từ fixture spike, `tests/fixtures/`):

| Nguồn                           | Dòng                                                                                                                                | Dùng cho                                                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| llmcompressor (tqdm, có `\r`) | `(N/33): Calibrating: x/512`, `(N/33): Propagating`                                                                              | tiến trình subgraph, ETA                                                                                                                      |
| llmcompressor (loguru)           | `\| compress_module_list \| INFO - Quantizing <module> using 512 samples`                                                            | module đang quantize                                                                                                                           |
| llmcompressor (loguru)           | `\| GPTQ \| METRIC - time 1.07s`, `METRIC - error 1152.89`                                                                         | thời gian / lỗi GPTQ mỗi module                                                                                                              |
| llmcompressor (loguru)           | `_apply_smoothing \| INFO - Smoothing with <norm>`                                                                                  | tiến trình SmoothQuant                                                                                                                        |
| vLLM                             | `Using MarlinLinearKernel for CompressedTensorsWNA16` / `Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8` | kernel (2 dạng: `Using X for Y`, `Selected X for Y`)                                                                                       |
| vLLM                             | `Model loading took 5.11 GiB memory and 11.19 seconds`, `GPU KV cache size: 104,448 tokens`                                      | bộ nhớ model, KV (regex bám nội dung: tiền tố `[file:line]` đổi theo model — `gpu_model_runner.py:5515` / `model_runner.py:380`) |
| vLLM                             | `No available memory for the cache blocks`                                                                                         | weight không vừa ngân sách VRAM                                                                                                             |
| llmcompressor                    | `Linearizing experts`, `MoE is being linearized after loading`                                                                   | bước tách expert MoE                                                                                                                         |
| vLLM                             | `Downcasting torch.float32 to torch.bfloat16.`                                                                                     | checkpoint lưu FP32                                                                                                                            |
| vLLM                             | `max_num_seqs (256) exceeds available Mamba cache blocks (124)`                                                                    | lỗi cấu hình hybrid, trích nguyên văn                                                                                                     |
| lm_eval                          | `Processed prompts: …/50`, bảng `\|Tasks\|Version\|Filter\|…`                                                                     | tiến trình, bảng in lại                                                                                                                     |

Log vLLM **không phân biệt W4A16 với W4A8** (spike Q4) → bằng chứng W4A8 là env truyền vào container + dòng `MarlinLinearKernel`.
nvidia-smi `--query-compute-apps` chỉ cho `process_name=python` → preflight map PID → container qua `/proc/<pid>/cgroup` (id container) + `docker ps`; không map được → in PID trần.

## 6. Luồng chạy

### `qtool quantize`

1. Validate tham số.
2. Lập kế hoạch: scheme → checkpoint cần có; kiểm tra cache checkpoint và cache baseline từng task.
3. In kế hoạch. `--dry-run` → exit 0.
4. Preflight (mục 7A). Image chưa có → tự build, in `[setup] building qtool-compress (first run only)`.
5. Mỗi checkpoint cần build → 1 container compress:
   - `[1/4]` đọc `config.json` (chặn model đã quantize **trước khi** load weight), load model.
   - `[2/4]` tập calib: lần đầu tải + áp chat template + tokenize + lưu `runs/<model>/calib/`; lần sau `reused from cache`.
   - `[3/4]` `oneshot` → (compute FP32: cast về dtype gốc) → `save_pretrained(save_compressed=True)` + processor + MTP
     vào `checkpoints/.tmp-<NAME>/` → thành công mới rename thành `checkpoints/<NAME>/` kèm `qtool.json`.
   - `oneshot` sửa model tại chỗ nên mỗi checkpoint load lại model gốc.
6. Trừ khi `--no-eval`: `[4/4]` baseline BF16 (bỏ qua nếu cache hợp lệ) → từng scheme, mỗi lần 1 container eval.
7. Bảng tổng hợp + `summary.json`.

### Cấu hình eval (ghim theo spike Q4)

```
lm_eval --model vllm \
  --model_args pretrained=<model>,max_model_len=4096,gpu_memory_utilization=0.85,max_num_seqs=64,enable_thinking=False[,quantization=fp8] \
  --tasks <tasks> --num_fewshot <mặc định của task> [--limit N] \
  --apply_chat_template --fewshot_as_multiturn --batch_size auto --output_path <tmp> --log_samples
```

- **Chế độ chat, tắt thinking** — lệch khỏi docs vLLM (completion, `add_bos_token=True`). Lý do: Qwen3.5 là model chat;
  BF16 gsm8k limit 50 ra 0.82 ở chế độ chat, 0.70 ở chế độ docs. Không truyền `add_bos_token` (Qwen không có BOS token).
- **Tự lùi về completion** (bỏ `--apply_chat_template --fewshot_as_multiturn`, in lý do) khi tokenizer không có chat template
  hoặc template làm rơi nội dung chuỗi (kiểm như calib, mục 3). Spike Q9: SmolVLM2 chế độ chat gửi prompt rỗng.
  Baseline và model quantize luôn cùng chế độ.
- `enable_thinking=False` truyền vô hại với template không có thinking (Phi, OLMoE, SmolVLM).
- `max_num_seqs=64` bắt buộc với model hybrid (Mamba/GDN): mặc định 256 vượt số Mamba cache block → vLLM từ chối khởi động.
- Khởi tạo vLLM mất ~4.5 phút **mỗi lượt eval** (load + compile + CUDA graph) → console in dòng chờ có thời gian trôi qua
  để dev biết tool không treo.

### Thời gian tham khảo (spike, rtx4070 trừ khi ghi khác)

| Việc                                              | Thời gian                                                                        |
| -------------------------------------------------- | --------------------------------------------------------------------------------- |
| Quantize 4B W4A16 BF16                             | ~15 phút (oneshot 906 s)                                                         |
| Quantize 4B W8A8 BF16                              | ~24 phút (SmoothQuant ~9 + GPTQ ~15)                                             |
| Quantize 9B W4A16 BF16                             | ~22 phút                                                                         |
| Quantize Phi-4-mini (3.8B) / SmolVLM2 (2.2B) W4A16 | ~10 / ~7 phút                                                                    |
| Quantize OLMoE 1B-7B (MoE 64 expert) W4A16         | ~50 phút                                                                         |
| Quantize 4B W4A16 FP32 trên V100                  | ~2 giờ 40 phút (10.6× 4070; forward FP32 chậm 12.5×, GPTQ 4.6×)             |
| Eval gsm8k limit 50                                | ~4.5 phút Qwen3.5 / ~2 phút model không hybrid (gần hết là khởi tạo vLLM) |
| Eval gsm8k full (1319 câu)                        | ~11 phút (ước)                                                                 |

ETA trong lúc quantize: tool tính từ `(N/33)` đã parse và thời gian đã trôi, in kèm nhãn `est.`.

### `qtool eval`

Các bước 1–4 như trên (validate, kế hoạch, in kế hoạch, preflight; không cần image compress),
rồi `[4/4]` cho baseline (nếu `--baseline`) và model, rồi bảng tổng hợp. Step 4 in thêm
`quantization_config` đọc từ `config.json` của model.

### Thư mục runs

```
~/qtool/runs/<model>/
├── calib/<key>/                  tập calib đã tokenize
├── checkpoints/INT4-W4A16/       + qtool.json
├── checkpoints/INT8-W8A8/        + qtool.json
├── checkpoints/NVFP4/            + qtool.json
├── results/bf16_<task>.json, <scheme>_<task>.json, summary.json
└── logs/<scheme>.<step>.log      log gốc đầy đủ
```

Kế hoạch in đủ mọi lượt, kể cả lượt không tạo checkpoint, để không scheme nào "biến mất":

```
Plan for Qwen/Qwen3.5-4B (3 schemes)
  quantize  INT4-W4A16  -> int4-w4a8-fp8
  quantize  NVFP4       -> nvfp4-w4a4
  runtime   fp8-w8a8    (no checkpoint, vLLM quantizes to FP8 at load time)
  eval      bf16 baseline (cache: none, will run)
  eval      int4-w4a8-fp8, nvfp4-w4a4, fp8-w8a8   tasks: gsm8k (full)
```

### Quy tắc cache

- Checkpoint dùng lại chỉ khi `qtool.json` khớp tham số lần chạy (model, recipe, calib source,
  samples, seq len, phiên bản llmcompressor, compute dtype). Không khớp → exit 2, in từng trường khác, gợi ý `--force`.
- Baseline dùng lại chỉ khi khớp task, limit, số shot, image eval. Không khớp → tự chạy lại, in lý do
  (`baseline cache stale: eval-limit 250 → full`).

### An toàn

- Ghi atomic (thư mục `.tmp-*` → rename khi thành công).
- Ctrl-C: `docker stop qtool-<pid>`, xoá thư mục tạm, in `Interrupted during [3/4] INT8-W8A8`, exit 130.
- Lock file `~/qtool/.lock` chặn 2 lượt chạy song song.

## 7. Xử lý lỗi

Phát hiện sớm nhất có thể; thông báo luôn trích dòng lỗi gốc + đường dẫn log + gợi ý. Không tự retry, không tự sửa.

### A. Preflight — exit 2, chưa chạy gì

| Tình huống                                                                                                                                                                                             | Phát hiện                               | Xử lý                                                                                                                                                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Không có Docker / GPU trong container                                                                                                                                                                  | `docker info`, `nvidia-smi`           | báo lỗi                                                                                                                                                                                                                     |
| VRAM trống < ngưỡng: BF16 0.6 × size, FP32 1.0 × size (spike Q3, Q6). **size = số tham số × 2 byte** (đọc header safetensors; model lưu FP32 như SmolVLM2 không bị tính gấp đôi) | nvidia-smi                                | liệt kê PID / container / VRAM đang chiếm, dev tự dừng                                                                                                                                                                  |
| RAM khả dụng < ngưỡng: BF16 1.6 × size, FP32 3.5 × size (spike: Qwen 4B +1.2×, Phi-4-mini +1.6×, FP32 4B +28.6 GB)                                                                               | `/proc/meminfo` `MemAvailable`        | báo số cần / số có; gợi ý GPU / máy khác                                                                                                                                                                             |
| Weight model eval (baseline BF16 hoặc checkpoint) > `gpu_memory_utilization × VRAM` trừ dư KV / CUDA graph                                                                                         | size safetensors, nvidia-smi              | bỏ lượt đó kèm lý do (`bf16 baseline does not fit: 13.8 GB weights > 13.6 GB budget`), Δ / recovery `n/a`; không để vLLM lỗi giữa chừng (spike Q9: OLMoE BF16 `No available memory for the cache blocks`) |
| `quantize` với `trust_remote_code` bắt buộc (`auto_map` mà transformers không có class gốc)                                                                                                 | `config.json`                           | exit 2: chưa hỗ trợ model cần remote code                                                                                                                                                                                 |
| qtool khác đang chạy                                                                                                                                                                                  | `.lock`                                 | `another qtool run (pid N) is active`                                                                                                                                                                                       |
| Task lm_eval không tồn tại                                                                                                                                                                            | hỏi danh sách task trong container eval | `unknown task '<x>'`                                                                                                                                                                                                        |
| `--act` với checkpoint không phải INT4 sym group-wise                                                                                                                                               | `quantization_config`                   | báo kiểu thực tế                                                                                                                                                                                                          |
| Máy có ≥ 2 GPU mà thiếu `--gpu`                                                                                                                                                                   | nvidia-smi                                | in bảng GPU (index, tên, SM, VRAM trống, process đang chiếm) để dev chọn                                                                                                                                              |
| Scheme ✗ trên SM của GPU (bảng mục 3)                                                                                                                                                               | `compute_cap`                           | báo lỗi + gợi ý scheme thay thế, vd `int8-w8a8 is not supported on SM 10.0. Use fp8-w8a8.`                                                                                                                             |
| Scheme ⚠ trên SM của GPU                                                                                                                                                                              | `compute_cap`                           | **cảnh báo, vẫn chạy**, vd `fp8-w8a8 on SM 8.6 runs as W8A16 (weight-only)`                                                                                                                                       |
| SM 7.0 mà không có `--no-eval`                                                                                                                                                                      | `compute_cap`                           | báo lỗi: eval không chạy được trên SM 7.0, thêm `--no-eval` rồi `qtool eval` trên GPU SM ≥ 7.5                                                                                                                |
| `--act` trên GPU không phải SM 8.9 / 12.0                                                                                                                                                           | `compute_cap`                           | báo lỗi                                                                                                                                                                                                                     |
| Cache checkpoint không khớp                                                                                                                                                                            | `qtool.json`                            | in diff, gợi ý `--force`                                                                                                                                                                                                  |

Preflight in thêm dung lượng đĩa trống (tham khảo, không chặn).

### B. Lúc chạy — scheme FAILED, xử lý tiếp theo `--on-error`

| Tình huống                                     | Xử lý                                                                                                   |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Model đã quantize khi `quantize`             | dừng ở [1/4] trước khi load weight, gợi ý `qtool eval`                                            |
| Model không phải BF16                          | cảnh báo, load đúng dtype gốc                                                                        |
| Kiến trúc không được hỗ trợ              | trích lỗi gốc + phiên bản thư viện trong image                                                     |
| `ImportError` gói phụ (vd `num2words`)     | trích nguyên văn, ghi rõ image nào thiếu gói                                                       |
| Calib rơi về text thô / token trung vị < 64  | cảnh báo ở `[2/4]`, vẫn chạy (đã in dạng template dùng)                                        |
| HF 401/403                                       | gợi ý `HF_TOKEN`; lỗi mạng khác: trích lỗi gốc                                                  |
| OOM khi quantize                                 | trích dòng OOM, gợi ý giảm `--calib-seq-len` / giải phóng GPU                                    |
| Container compress exit 137 (chạm `--memory`) | báo hết RAM host + số RAM đã cấp; log gốc dừng đột ngột nên không có dòng lỗi để trích |
| OOM khi vLLM load                                | trích lỗi gốc (`max_model_len` ghim sẵn theo spike)                                                 |
| Hết đĩa (`ENOSPC`)                          | xoá thư mục tạm, báo lỗi                                                                            |

Luôn dừng toàn bộ bất kể `--on-error`: lỗi bước dùng chung (tập calib). Lỗi checkpoint `INT4-W4A16` → mọi scheme
dùng nó (`int4-w4a16`, `int4-w4a8-*`) SKIPPED kèm lý do. Lỗi checkpoint `NVFP4` chỉ ảnh hưởng `nvfp4-w4a4`.

### C. Parse

| Tình huống                                        | Xử lý                                                                              |
| --------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Thiếu dòng mong đợi                             | `unknown (not found in <nguồn>)`                                                  |
| `--act` mà log không có `MarlinLinearKernel` | kết quả **INVALID**: vẫn in điểm, ghi rõ không chứng minh được W4A8 |
| Không có `results.json`                         | FAILED, không in điểm                                                             |

## 8. Console output

Minh hoạ (bố cục chốt; ignore list, số subgraph, kích thước, VRAM lấy từ spike; điểm eval full là giả):

```
$ qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16,int4-w4a8-fp8

qtool 0.1.0 · 2026-10-30 21:00:03 · RTX 4070 Ti SUPER 16 GB SM 8.9
[preflight] GPU 0 · SM 8.9 · VRAM free 15.9 / 16.0 GB (need 5.3) · RAM avail 27.1 GB (need 12.3)
[preflight] compute dtype bfloat16 · image qtool-compress:cu129 · disk free 659 GB

Plan for Qwen/Qwen3.5-4B (2 schemes)
  quantize  INT4-W4A16  -> int4-w4a16, int4-w4a8-fp8
  eval      bf16 baseline (cache: none, will run)
  eval      int4-w4a16, int4-w4a8-fp8   tasks: gsm8k (full)

══════ [1/4] Loading model · INT4-W4A16 ═══════════════════ 21:00:09
  source       : Qwen/Qwen3.5-4B (HF cache hit)
  architecture : Qwen3_5ForConditionalGeneration (verified) · multimodal · hybrid · MTP
  dtype        : bfloat16 · quantized: no
  ✓ done in 38s

══════ [2/4] Preparing calibration data ═══════════════════ 21:00:47
  source       : HuggingFaceH4/ultrachat_200k split=train_sft · 512 samples · seed 42
  template     : chat template (string content)
  tokens/sample: min 199 · median 1,080 · max 2,048 (truncated: 48)
  ✓ done in 10s

══════ [3/4] Quantizing INT4-W4A16 ═════════════════════════ 21:01:08
  recipe       : GPTQModifier targets=Linear scheme=W4A16 · sequential Qwen3_5DecoderLayer
  ignore       : lm_head (output head) · visual (vision) · linear_attn (hybrid mixer)
  quantized    : 128 Linear (mlp.*, self_attn.*)
  (17/33) Calibrating 512/512 [00:13] · elapsed 7m 41s · est. 7m 15s left
  saved        : checkpoints/INT4-W4A16 · 8.8 GB → 5.2 GB · GPU peak 5.0 GB · MTP saved
  ✓ done in 15m 12s

══════ [4/4] Evaluating int4-w4a8-fp8 ══════════════════════ 21:16:20
  checkpoint   : checkpoints/INT4-W4A16 (shared)
  quantization : compressed-tensors · INT4 W4A16 g128 symmetric
  runtime env  : VLLM_MARLIN_INPUT_DTYPE=fp8
  vLLM         : starting (load + compile + CUDA graphs) … 3m 52s
  kernel       : MarlinLinearKernel for CompressedTensorsWNA16
  model memory : 5.11 GiB · KV cache 104,448 tokens
  lm_eval      : 1319/1319 [06:12]
  |Tasks|Filter          |n-shot|Metric     |Value|Stderr|
  |gsm8k|strict-match    |     5|exact_match|0.784|±0.011|
  |gsm8k|flexible-extract|     5|exact_match|0.796|±0.011|
  vs bf16      : strict-match 0.812 → 0.784 · −2.8 pp · recovery 96.6%
  ✓ done in 6m 58s

══════ Summary · gsm8k strict-match (full) ═════════════════ total 41m 10s
  scheme          status   accuracy        Δ vs bf16   recovery   size
  bf16            OK       0.812 ± 0.011   —           —          8.8 GB
  int4-w4a16      OK       0.792 ± 0.011   −2.0 pp     97.5%      5.2 GB
  int4-w4a8-fp8   OK       0.784 ± 0.011   −2.8 pp     96.6%      5.2 GB (shared)
  results: ~/qtool/runs/Qwen3.5-4B/results/summary.json
```

Định nghĩa: Δ pp = quant − bf16 (điểm phần trăm); recovery % = quant / bf16 × 100 (cách RedHat dùng
trên model card). Bảng `|Tasks|...|` in lại nguyên văn từ lm_eval.

Step 4 của `nvfp4-w4a4` in `quantization : compressed-tensors · NVFP4 W4A4 g16` (đọc `quantization_config`) và dòng kernel
(SM 8.9: `MarlinNvFp4LinearKernel for NVFP4`; log gốc là `... for NVFP4 GEMM`, `KERNEL_RE` giữ 1 từ sau `for`). Step 3 của NVFP4 không có dòng `GPTQ | METRIC` →
chỉ in tiến trình subgraph; dòng nào không thấy thì theo quy tắc `unknown`, không đoán.

## 9. Testing (TDD)

| Tầng          | Chạy ở                                        | Nội dung                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Unit        | local, không GPU/Docker                        | `cli`, `schemes`, `plan` (gộp, cache, stale, dry-run), `parsers` trên **fixture log thật từ spike** (cả trường hợp thiếu dòng), `preflight` trên CSV nvidia-smi (chọn GPU / bắt buộc `--gpu`, bảng khả năng SM ✓/⚠/✗, `--compute-dtype auto`, ngưỡng VRAM), `report`, `runner` với docker giả (stub phát lại fixture, exit code tuỳ chọn, giả lập SIGINT): ghi atomic, dọn tạm, `--on-error`, exit 0/1/130 |
| 2. Container   | image compress, CPU, model tí hon ngẫu nhiên | `recipe.py` trên model tí hon cấu hình ngẫu nhiên của **từng kiến trúc đã kiểm chứng** (Qwen3.5, Phi3, OLMoE, SmolVLM): ignore list, layer tuần tự, số `Linear` quantize, linearize MoE, kiểm template (có tokenizer rơi nội dung); `quantize.py`: thống kê calib, dòng `@qtool`, chặn model đã quantize, `oneshot` đầu-cuối, file copy từ nguồn, `qtool.json`                                                     |
| 3. Integration | rtx4070, `@pytest.mark.gpu`, chạy tay        | `qtool quantize` đầu-cuối với Qwen3.5 nhỏ nhất, `--eval-limit 20`                                                                                                                                                                                                                                                                                                                                                                                          |
| 4. Acceptance  | rtx4070                                         | `qtool eval RedHatAI/Qwen3.5-4B-quantized.w4a16 --act fp8` ra đúng dòng kernel như log bake-off 26/10; `int4-w4a16` tự làm (cùng recipe RedHat) nằm trong dải dao động của script tham chiếu RedHat chạy trên cùng image, gsm8k full (GPTQ không tất định giữa các lần, ~±2 pp; xem `docs/spike/2026-11-01-acceptance-followup.md`)                                                                                                    |

Bổ sung bản 3.2:

| Tầng         | Nội dung                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Unit       | bảng scheme 6 tên + 3 nhóm; `checkpoints_for` (NVFP4 riêng; `int4-w4a8-*` dùng chung `INT4-W4A16`); hàng `nvfp4-w4a4` của bảng SM (⚠ 8.9, ✓ 12.x unverified, ✗ 7.0) và cảnh báo preflight; `--scheme int4-w4a8-fp8` một mình → build `INT4-W4A16`, chỉ 1 lượt eval; `--help` và `unknown scheme` theo 3 nhóm; plan in dòng `runtime`; `quantization_text` cho NVFP4; parser dòng `MarlinNvFp4LinearKernel` (fixture thật từ lượt acceptance)                                                                                                                                                                                                                                                                        |
| 2. Container  | `quantize.py` với `NVFP4` trên model tí hon (CPU): recipe in ra, checkpoint có `quantization_config` NVFP4, `qtool.json`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 4. Acceptance | rtx4070, Qwen3.5-4B, gsm8k full: (a) `qtool quantize --scheme nvfp4-w4a4` load được trong vLLM 0.28 với `MarlinNvFp4LinearKernel`; (b) `qtool eval AxionML/Qwen3.5-4B-NVFP4` (checkpoint NVFP4 dùng trong bake-off) trên cùng image làm mốc — không đặt ngưỡng cứng vì recipe của họ không công bố, chỉ báo chênh lệch; (c) chạy thêm 1 lần NVFP4 bằng `GPTQModifier(scheme="NVFP4")` (thử tay, không phải scheme): hơn RTN > 2 pp (mức dao động GPTQ giữa các lần, xem acceptance gốc ở trên) → đổi recipe mặc định sang GPTQ và ghi lại; không thì giữ `QuantizationModifier`; (d) `--scheme int4-w4a8-fp8` dùng lại checkpoint `INT4-W4A16` cũ (không quantize lại) và ra đúng 1 hàng |
| 4. Kết quả 02/11 | `docs/spike/2026-11-02-nvfp4-acceptance.md`: (a) qtool NVFP4 0.804 vs BF16 0.805 (−0.1 pp), 5.7 GB, quantize 8m49s; (b) AxionML 0.763 (−4.7 pp, quantize cả Linear của GDN), 4.2 GB; (c) bỏ: RTN đã 99.9 %; (d) bỏ: code giống `main`, đã có unit test |

Output test phải sạch.

## 10. Spike (đã xong 2026-10-30)

Kết quả: `docs/spike/2026-10-30-findings.md` + `tests/fixtures/`. Cả 8 câu có câu trả lời từ lượt chạy thật;
các mục 3, 5, 6, 7 của spec bản 2 đã cập nhật theo đó. Spike 2 (2026-10-31, findings Q9): recipe chung
cho mọi kiến trúc, thử trên Phi-4-mini, OLMoE, SmolVLM2. Danh sách câu hỏi gốc:

1. Tìm phiên bản llmcompressor hiểu Qwen3.5 **và** checkpoint xuất ra load được bằng vLLM 0.28 → ghim.
   Không có cặp nào → dừng, quay lại bàn kiến trúc (đổi image eval ảnh hưởng khả năng so với số bake-off).
2. Đọc `config.json` Qwen3.5-4B + ignore list trong checkpoint RedHat → chốt ignore list từng recipe.
3. Đo VRAM đỉnh quantize 4B (và 9B nếu vừa), BF16 và FP32 → hệ số theo kích thước model → công thức ngưỡng preflight.
4. Chốt `max_model_len`, `gpu_memory_utilization` cho eval; kiểm `add_bos_token=True` (docs vLLM khuyên) có hợp với tokenizer Qwen không.
5. Thử `VLLM_MARLIN_INPUT_DTYPE=int8` trên vLLM 0.28.
6. Quantize trên GPU SM < 8.0: image compress có torch kèm SM 7.0 (CUDA 12.x) **và** llmcompressor
   tương thích torch đó; chạy GPTQ FP32 với model 4B trên 1 GPU V100 trống của Apollo (hỏi Nhân
   GPU nào trước khi dùng); checkpoint xuất ra phải load + eval được trên rtx4070.
   Không có tổ hợp phiên bản nào chạy được → bỏ hỗ trợ SM < 8.0 khỏi bản đầu, báo lại Nhân.
7. Kiểm vLLM 0.28 còn build SM 7.5 không → chốt cột SM 7.5 bảng mục 3.
8. Thu log thật: llmcompressor (GPTQ, SmoothQuant), vLLM (có / không `--act`), lm_eval (tiến trình, bảng, `results.json`, sample) → `tests/fixtures/`.

## 11. Thay đổi bản 3.2 (2026-11-02)

- **Thêm `nvfp4-w4a4`** (checkpoint `NVFP4`, `QuantizationModifier` preset NVFP4). Lý do: chuẩn bị cho GPU Blackwell, nơi
  FP4×FP4 chạy native; trên GPU cũ hơn chạy được nhưng chỉ weight-only (cảnh báo). Không cần đổi image.
- **Scheme chia 3 nhóm theo checkpoint** trong spec, README, `--help`, thông báo lỗi và kế hoạch: tạo checkpoint
  (`int4-w4a16`, `int8-w8a8`, `nvfp4-w4a4`) / dùng lại `INT4-W4A16` (`int4-w4a8-*`) / không checkpoint (`fp8-w8a8`).
  Lý do: bảng cũ đặt 5 scheme ngang hàng khiến người dùng tưởng W4A8 phải quantize riêng.
- Đã cân nhắc và **không làm**: thay `int4-w4a8-*` bằng cờ `quantize --act` (buộc eval thêm W4A16 khi chỉ cần W4A8);
  bỏ `fp8-w8a8` khỏi `quantize` (mất bảng so sánh chung một lượt).
- Ngoài phạm vi ghi rõ thêm: NVFP4A16, MXFP4, INT4 W4A4, INT4 bất đối xứng / AWQ (mục 2).
