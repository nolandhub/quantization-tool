# qtool — Design Spec

Ngày: 2026-10-30 · Trạng thái: chờ Nhân duyệt · Repo: `~/ws-DoNhan/qtool` (local) · máy phát triển và kiểm chứng: `rtx4070:~/qtool` (RTX 4070 Ti SUPER, SM 8.9)

## 1. Mục tiêu

CLI đơn giản gói 4 bước quantize theo flow LLM Compressor của vLLM
(<https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/>):

1. Loading the model
2. Preparing calibration data
3. Applying quantization
4. Evaluating accuracy in vLLM

Dev chỉ cần chỉ tên model và scheme. Console log bằng tiếng Anh, từng bước in rõ đang làm gì,
tiến trình tới đâu, mất bao lâu. **Parse, không tự chế:** mọi tiến trình và số liệu hiển thị lấy
từ output thật của llmcompressor / vLLM / lm_eval hoặc từ file (config.json, results.json);
tool chỉ tự đo thời gian và tự tính Δ pp, recovery % từ các số đã parse.

Lý do phải tự làm: vLLM/LLM Compressor chỉ cung cấp từng mảnh — `llmcompressor.oneshot()`
(Python API, không có CLI quantize; console script duy nhất là `llmcompressor.trace` và
`reindex_fused_weights` đã deprecated), `lm_eval` (CLI riêng, không phải dependency của
llmcompressor), cờ runtime `VLLM_MARLIN_INPUT_DTYPE` của vLLM. qtool là lớp keo nối các mảnh.

## 2. Phạm vi

### Trong phạm vi

- 2 lệnh con: `qtool quantize`, `qtool eval`.
- 5 scheme (bảng mục 3), chỉ sinh scheme được yêu cầu.
- Phần cứng: GPU NVIDIA bất kỳ có Docker + NVIDIA Container Toolkit; 1 GPU mỗi lượt chạy, chọn bằng
  `--gpu` (bắt buộc khi máy có ≥ 2 GPU). Scheme nào eval được trên GPU nào: bảng mục 3.
- Eval bằng `lm_eval --model vllm`, task bất kỳ của lm_eval, mặc định `gsm8k`.

### Ngoài phạm vi (non-goals)

- Không dừng / bật container hay service nào. Dev tự làm; tool chỉ kiểm tra và báo.
- Không tích hợp qbench, voice, rag hay bất cứ gì của `qwen-bench/`.
- Không MXFP4, NVFP4, W8A16, recipe mixed-precision, checkpoint FP8 lưu sẵn.
- Không multi-GPU / tensor parallel.
- Không eval trên SM 7.0 (V100): vLLM 0.28 bỏ SM 7.0, vLLM 0.18 (bản cuối có SM 7.0) không biết
  Qwen3.5. V100 chỉ quantize (`--no-eval`), eval ở máy khác.
- Không kiểm định thống kê (McNemar, p-value). Theo cách làm gốc: bảng lm_eval có stderr + recovery %.

### Hướng mở rộng (ghi lại, chưa làm)

- Preset mixed-precision theo nhóm layer (giữ attention/GDN 8 bit, MLP 4 bit), kiểu `RedHatAI/GLM-5.3-MXFP4xFP8`.
- Tập calib tiếng Việt hoặc trộn (`--calib-mix`).
- Recipe `FP8_DYNAMIC` lưu checkpoint.
- Multi-GPU (tensor parallel) cho model không vừa 1 GPU.

## 3. Scheme

| `--scheme` | Weight | Activation | Checkpoint | Eval runtime |
|---|---|---|---|---|
| `int4-w4a16` | INT4 sym g128 | BF16 | build `INT4-W4A16`: `GPTQModifier(targets="Linear", scheme="W4A16")` | mặc định |
| `int4-w4a8-fp8` | INT4 | FP8 | dùng chung `INT4-W4A16` | `VLLM_MARLIN_INPUT_DTYPE=fp8` |
| `int4-w4a8-int8` | INT4 | INT8 | dùng chung `INT4-W4A16` | `VLLM_MARLIN_INPUT_DTYPE=int8` — experimental, chưa kiểm chứng |
| `int8-w8a8` | INT8 | INT8 | build `INT8-W8A8`: `SmoothQuantModifier(smoothing_strength=0.8)` + `GPTQModifier(targets="Linear", scheme="W8A8")` | mặc định |
| `fp8-w8a8` | FP8 | FP8 | không có | model BF16 gốc + `--quantization fp8` (vLLM quantize online) |

Ghi chú:

- W4A8 trên Ada = checkpoint INT4 đối xứng g128 + Marlin lượng tử hoá activation động lúc chạy
  (đã kiểm chứng trong bake-off 26/10, `qwen-bench/docs/handoff-2026-10-27.md`). Checkpoint gắn
  nhãn `CompressedTensorsW4A8Fp8` đòi SM 9.0 nên không dùng.
- Ignore list mỗi recipe: `lm_head` + các layer xác định từ spike (mục 10). Không đoán.
- Quantize luôn đi từ model gốc chưa quantize.

### Phần cứng

Nguyên tắc:

- **Quantize (step 1–3) chạy đúng trên mọi GPU.** Phần forward calib cần một kiểu số biểu diễn được
  model gốc không tràn: BF16 hoặc FP32. Phần toán GPTQ / SmoothQuant vốn tính FP32. Kiểu đích
  (INT4 / INT8 / FP8) chỉ là phép làm tròn, không cần phần cứng hỗ trợ. Tensor core BF16 (SM ≥ 8.0)
  chỉ giúp nhanh hơn; không có thì tính FP32 trên CUDA core (chậm hơn, tốn gấp đôi bộ nhớ).
  Không bao giờ tự chọn FP16 (model train BF16 có thể tràn FP16 → calib sai ngầm).
- **Eval (step 4) cần kernel vLLM cho đúng định dạng** → giới hạn theo SM (bảng dưới).
- **So với baseline** chỉ hợp lệ khi cùng GPU, cùng image eval, cùng task / limit / số shot (mục 6, quy tắc cache).

Bảng khả năng eval theo compute capability. ✓ = native, ⚠ = chạy nhưng khác bản chất (cảnh báo),
✗ = không chạy (preflight chặn). Chỉ cột SM 8.9 đã kiểm chứng trên phần cứng thật (rtx4070); các cột
khác theo docs vLLM + code vLLM 0.28 → tool in nhãn `[per vLLM docs, unverified]`.

| Scheme \ SM | 7.0 (V100) | 7.5 (T4, 20xx) | 8.0 / 8.6 (A100, 30xx) | **8.9 (L4, 40xx)** | 9.0 (H100) | 10.0 (B200) | 12.0 (50xx) |
|---|---|---|---|---|---|---|---|
| `int4-w4a16` | ✗ | ✗ (docs: INT4 cần > 8.0) | ✓ | ✓ | ✓ | ✓ | ✓ |
| `int4-w4a8-*` | ✗ | ✗ | ✗ | ✓ | ✗ (Marlin W4A8 chậm hơn W4A16) | ✗ | ✓ |
| `int8-w8a8` | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ (docs: không INT8 ≥ 10.0) | ✗ |
| `fp8-w8a8` | ✗ | ✗ | ⚠ chạy dạng W8A16 weight-only (Marlin) | ✓ | ✓ | ✓ | ✓ |
| baseline BF16 | ✗ | ⚠ vLLM ép FP16 | ✓ | ✓ | ✓ | ✓ | ✓ |
| quantize (step 1–3) | ⚠ FP32, chậm | ⚠ FP32, chậm | ✓ BF16 | ✓ BF16 | ✓ BF16 | ✓ BF16 | ✓ BF16 |

Hàng `baseline BF16` ở SM 7.5 và mọi ô SM 7.5 / 7.0 cần xác nhận lại trong spike hoặc khi có máy
(vLLM 0.28 có còn build SM 7.5 không).

## 4. CLI

### `qtool quantize MODEL --scheme SCHEMES [options]`

```
positional:
  MODEL                    HF repo id hoặc thư mục local của model gốc (vd Qwen/Qwen3.5-4B)

required:
  --scheme SCHEMES         phân tách bằng dấu phẩy: int4-w4a16, int4-w4a8-fp8, int4-w4a8-int8,
                           int8-w8a8, fp8-w8a8

calibration (step 2):
  --calib SOURCE           HF dataset id hoặc file .jsonl     [default: HuggingFaceH4/ultrachat_200k]
  --calib-samples N        số mẫu                              [default: 512]
  --calib-seq-len N        số token tối đa mỗi mẫu             [default: 2048]

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

### Exit code

| Code | Nghĩa |
|---|---|
| 0 | mọi scheme OK |
| 1 | ≥ 1 scheme FAILED (`--on-error continue`) |
| 2 | lỗi tham số / preflight, chưa chạy gì |
| 130 | Ctrl-C |

## 5. Kiến trúc

CLI Python trên host (chỉ thư viện chuẩn) điều khiển 2 Docker image; mỗi bước nặng chạy trong
một container `--rm` tên `qtool-<pid>`. stdout/stderr container được đọc từng dòng: ghi nguyên
văn vào file log, parse, in dòng đã lọc ra console.

```
qtool (host)
  ├─ docker run qtool-compress  → step 1–3 (transformers + llmcompressor, phiên bản ghim từ spike)
  └─ docker run qtool-eval      → step 4 (FROM vllm/vllm-openai:v0.28.0-cu129-ubuntu2404 + lm-eval[api], ghim)
```

Lý do 2 image: docs vLLM khuyên tách môi trường llmcompressor và vLLM; dùng đúng image vLLM 0.28
của bake-off để số eval so được với số cũ.

GPU: `docker run --gpus device=N` → container chỉ thấy đúng 1 GPU đã chọn (không thể bị rải sang
card khác). Mount: `--hf-cache` vào cả 2 container; `HF_TOKEN` truyền qua nếu có.
Image compress phải có torch build kèm SM 7.0 (CUDA 12.x; CUDA 13 bỏ Volta) để quantize được trên V100.

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
│   └── report.py       bảng tổng hợp, Δ pp, recovery %, summary.json
├── images/
│   ├── compress/Dockerfile
│   ├── compress/quantize.py   chạy trong container: load → calib → oneshot → save
│   └── eval/Dockerfile
├── tests/
├── docs/specs/
├── sync.sh             rsync local ↔ rtx4070:~/qtool
└── README.md
```

### Nguồn thông tin hiển thị

| Loại | Cách lấy |
|---|---|
| Sự kiện của thư viện (tqdm llmcompressor, dòng GPTQ, dòng kernel vLLM, tiến trình lm_eval) | regex trên output gốc; regex viết từ fixture log thật của spike |
| Sự kiện script mình đọc trong container (config.json, thống kê token calib, dung lượng checkpoint, VRAM đỉnh) | `quantize.py` in dòng `@qtool {json}`, host parse JSON |
| Kết quả eval | `results.json` + file sample của lm_eval |
| Thời gian, Δ pp, recovery % | host tự đo / tự tính từ số đã parse |

Không tìm thấy dòng mong đợi → in `unknown (not found in <nguồn>)`, không bao giờ đoán.

## 6. Luồng chạy

### `qtool quantize`

1. Validate tham số.
2. Lập kế hoạch: scheme → checkpoint cần có; kiểm tra cache checkpoint và cache baseline từng task.
3. In kế hoạch. `--dry-run` → exit 0.
4. Preflight (mục 7A). Image chưa có → tự build, in `[setup] building qtool-compress (first run only)`.
5. Mỗi checkpoint cần build → 1 container compress:
   - `[1/4]` đọc `config.json` (chặn model đã quantize **trước khi** load weight), load model.
   - `[2/4]` tập calib: lần đầu tải + áp chat template + tokenize + lưu `runs/<model>/calib/`; lần sau `reused from cache`.
   - `[3/4]` `oneshot` → ghi `checkpoints/.tmp-<NAME>/` → thành công mới rename thành `checkpoints/<NAME>/` kèm `qtool.json`.
   - `oneshot` sửa model tại chỗ nên mỗi checkpoint load lại model gốc.
6. Trừ khi `--no-eval`: `[4/4]` baseline BF16 (bỏ qua nếu cache hợp lệ) → từng scheme, mỗi lần 1 container eval.
7. Bảng tổng hợp + `summary.json`.

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
├── results/bf16_<task>.json, <scheme>_<task>.json, summary.json
└── logs/<scheme>.<step>.log      log gốc đầy đủ
```

### Quy tắc cache

- Checkpoint dùng lại chỉ khi `qtool.json` khớp tham số lần chạy (model, recipe, calib source,
  samples, seq len, phiên bản llmcompressor). Không khớp → exit 2, in từng trường khác, gợi ý `--force`.
- Baseline dùng lại chỉ khi khớp task, limit, số shot, image eval. Không khớp → tự chạy lại, in lý do
  (`baseline cache stale: eval-limit 250 → full`).

### An toàn

- Ghi atomic (thư mục `.tmp-*` → rename khi thành công).
- Ctrl-C: `docker stop qtool-<pid>`, xoá thư mục tạm, in `Interrupted during [3/4] INT8-W8A8`, exit 130.
- Lock file `~/qtool/.lock` chặn 2 lượt chạy song song.

## 7. Xử lý lỗi

Phát hiện sớm nhất có thể; thông báo luôn trích dòng lỗi gốc + đường dẫn log + gợi ý. Không tự retry, không tự sửa.

### A. Preflight — exit 2, chưa chạy gì

| Tình huống | Phát hiện | Xử lý |
|---|---|---|
| Không có Docker / GPU trong container | `docker info`, `nvidia-smi` | báo lỗi |
| VRAM trống < ngưỡng (tính theo kích thước model × hệ số đo từ spike, nhân đôi khi FP32) | nvidia-smi | liệt kê PID / process / VRAM đang chiếm, dev tự dừng |
| qtool khác đang chạy | `.lock` | `another qtool run (pid N) is active` |
| Task lm_eval không tồn tại | hỏi danh sách task trong container eval | `unknown task '<x>'` |
| `--act` với checkpoint không phải INT4 sym group-wise | `quantization_config` | báo kiểu thực tế |
| Máy có ≥ 2 GPU mà thiếu `--gpu` | nvidia-smi | in bảng GPU (index, tên, SM, VRAM trống, process đang chiếm) để dev chọn |
| Scheme ✗ trên SM của GPU (bảng mục 3) | `compute_cap` | báo lỗi + gợi ý scheme thay thế, vd `int8-w8a8 is not supported on SM 10.0. Use fp8-w8a8.` |
| Scheme ⚠ trên SM của GPU | `compute_cap` | **cảnh báo, vẫn chạy**, vd `fp8-w8a8 on SM 8.6 runs as W8A16 (weight-only)` |
| SM 7.0 mà không có `--no-eval` | `compute_cap` | báo lỗi: eval không chạy được trên SM 7.0, thêm `--no-eval` rồi `qtool eval` trên GPU SM ≥ 7.5 |
| `--act` trên GPU không phải SM 8.9 / 12.0 | `compute_cap` | báo lỗi |
| Cache checkpoint không khớp | `qtool.json` | in diff, gợi ý `--force` |

Preflight in thêm dung lượng đĩa trống (tham khảo, không chặn).

### B. Lúc chạy — scheme FAILED, xử lý tiếp theo `--on-error`

| Tình huống | Xử lý |
|---|---|
| Model đã quantize khi `quantize` | dừng ở [1/4] trước khi load weight, gợi ý `qtool eval` |
| Model không phải BF16 | cảnh báo, load đúng dtype gốc |
| Kiến trúc không được hỗ trợ | trích lỗi gốc + phiên bản thư viện trong image |
| HF 401/403 | gợi ý `HF_TOKEN`; lỗi mạng khác: trích lỗi gốc |
| OOM khi quantize | trích dòng OOM, gợi ý giảm `--calib-seq-len` / giải phóng GPU |
| OOM khi vLLM load | trích lỗi gốc (`max_model_len` ghim sẵn theo spike) |
| Hết đĩa (`ENOSPC`) | xoá thư mục tạm, báo lỗi |

Luôn dừng toàn bộ bất kể `--on-error`: lỗi bước dùng chung (tập calib); lỗi checkpoint
`INT4-W4A16` → các scheme `int4-w4a8-*` bị skip kèm lý do.

### C. Parse

| Tình huống | Xử lý |
|---|---|
| Thiếu dòng mong đợi | `unknown (not found in <nguồn>)` |
| `--act` mà log không có `MarlinLinearKernel` | kết quả **INVALID**: vẫn in điểm, ghi rõ không chứng minh được W4A8 |
| `--act int8` | nhãn `[experimental: not yet verified on this stack]` |
| Không có `results.json` | FAILED, không in điểm |

## 8. Console output

Minh hoạ (số liệu giả, ignore list và số layer chờ spike):

```
$ qtool quantize Qwen/Qwen3.5-4B --scheme int4-w4a16,int4-w4a8-fp8

qtool 0.1.0 · 2026-10-30 21:00:03 · RTX 4070 Ti SUPER 16 GB SM 8.9
[preflight] GPU 0 · SM 8.9 · free 15.8 / 16.0 GB · compute dtype bfloat16 · disk free 789 GB · images OK

Plan for Qwen/Qwen3.5-4B (2 schemes)
  quantize  INT4-W4A16  -> shared by: int4-w4a16, int4-w4a8-fp8
  eval      bf16 baseline (cache: none, will run)
  eval      int4-w4a16, int4-w4a8-fp8   tasks: gsm8k (full)

══════ [1/4] Loading model · INT4-W4A16 ═══════════════════ 21:00:09
  source       : Qwen/Qwen3.5-4B (HF cache hit)
  architecture : Qwen3_5ForConditionalGeneration
  dtype        : bfloat16 · quantized: no
  ✓ done in 38s

══════ [2/4] Preparing calibration data ═══════════════════ 21:00:47
  source       : HuggingFaceH4/ultrachat_200k split=train_sft · 512 samples · seed 42
  tokens/sample: min 61 · median 1,184 · max 2,048 (truncated: 97)
  ✓ done in 21s

══════ [3/4] Quantizing INT4-W4A16 ═════════════════════════ 21:01:08
  recipe       : GPTQModifier targets=Linear scheme=W4A16
  ignore       : lm_head, <từ spike>
  (17/33) Calibrating 512/512 [00:13]
  saved        : checkpoints/INT4-W4A16 · 8.4 GB → 3.1 GB · GPU peak 11.2 GB
  ✓ done in 8m 47s

══════ [4/4] Evaluating int4-w4a8-fp8 ══════════════════════ 21:17:53
  checkpoint   : checkpoints/INT4-W4A16 (shared)
  quantization : compressed-tensors · INT4 W4A16 g128 symmetric
  runtime env  : VLLM_MARLIN_INPUT_DTYPE=fp8
  kernel       : MarlinLinearKernel for CompressedTensorsWNA16
  lm_eval      : 1319/1319 [06:12]
  |Tasks|Filter          |n-shot|Metric     |Value|Stderr|
  |gsm8k|strict-match    |     5|exact_match|0.784|±0.011|
  |gsm8k|flexible-extract|     5|exact_match|0.796|±0.011|
  vs bf16      : strict-match 0.812 → 0.784 · −2.8 pp · recovery 96.6%
  ✓ done in 6m 58s

══════ Summary · gsm8k strict-match (full) ═════════════════ total 41m 10s
  scheme          status   accuracy        Δ vs bf16   recovery   size
  bf16            OK       0.812 ± 0.011   —           —          8.4 GB
  int4-w4a16      OK       0.792 ± 0.011   −2.0 pp     97.5%      3.1 GB
  int4-w4a8-fp8   OK       0.784 ± 0.011   −2.8 pp     96.6%      3.1 GB (shared)
  results: ~/qtool/runs/Qwen3.5-4B/results/summary.json
```

Định nghĩa: Δ pp = quant − bf16 (điểm phần trăm); recovery % = quant / bf16 × 100 (cách RedHat dùng
trên model card). Bảng `|Tasks|...|` in lại nguyên văn từ lm_eval.

## 9. Testing (TDD)

| Tầng | Chạy ở | Nội dung |
|---|---|---|
| 1. Unit | local, không GPU/Docker | `cli`, `schemes`, `plan` (gộp, cache, stale, dry-run), `parsers` trên **fixture log thật từ spike** (cả trường hợp thiếu dòng), `preflight` trên CSV nvidia-smi (chọn GPU / bắt buộc `--gpu`, bảng khả năng SM ✓/⚠/✗, `--compute-dtype auto`, ngưỡng VRAM), `report`, `runner` với docker giả (stub phát lại fixture, exit code tuỳ chọn, giả lập SIGINT): ghi atomic, dọn tạm, `--on-error`, exit 0/1/130 |
| 2. Container | image compress, CPU, model tí hon ngẫu nhiên | `quantize.py`: thống kê calib, dòng `@qtool`, chặn model đã quantize, `oneshot` đầu-cuối, `qtool.json` |
| 3. Integration | rtx4070, `@pytest.mark.gpu`, chạy tay | `qtool quantize` đầu-cuối với Qwen3.5 nhỏ nhất, `--eval-limit 20` |
| 4. Acceptance | rtx4070 | `qtool eval RedHatAI/Qwen3.5-4B-quantized.w4a16 --act fp8` ra đúng dòng kernel như log bake-off 26/10; `int4-w4a16` tự làm vs checkpoint RedHat trên gsm8k full nằm trong stderr |

Output test phải sạch.

## 10. Spike (làm trước khi code)

Kết quả là câu trả lời + fixture; code spike bỏ đi. Dev tự dừng `qwen3.5-9b-fp8` trong lúc spike.

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
