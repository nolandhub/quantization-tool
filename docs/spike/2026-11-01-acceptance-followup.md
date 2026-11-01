# Acceptance v0.1 — thí nghiệm tìm nguyên nhân lệch với RedHat (2026-11-01, rtx4070)

Bối cảnh: `docs/spike/2026-10-30-findings.md` mục "Acceptance v0.1". gsm8k full (1319 câu), chat mode, cùng image eval;
so cặp từng câu bằng McNemar chính xác. Eval full tất định (RedHat chạy 2 lần: 0 câu lệch).

| Checkpoint (INT4-W4A16) | calib | dampening | strict | flexible | vs RedHat (strict) |
|---|---|---|---|---|---|
| RedHat | Open-Platypus 1024×8192, text | 0.05 | 0.8105 | 0.8249 | — |
| qtool mặc định | ultrachat 512×2048, chat | 0.01 | 0.7870 | 0.8044 | 99/68, p = 0.020 |
| qtool + calib RedHat | Open-Platypus 1024×8192, text | 0.01 | 0.7900 | 0.8074 | 89/62, p = 0.034 |
| qtool + dampening 0.05 | ultrachat 512×2048, chat | **0.05** | **0.8059** | **0.8211** | 84/78, p = 0.69 |

- Dữ liệu calib **không** phải nguyên nhân (mặc định vs Platypus: p = 0.82; Platypus median chỉ 53 token/mẫu).
- **`dampening_frac 0.05` đóng khoảng cách**: còn −0.46 pp so với RedHat, không phân biệt được (p = 0.69); so với qtool
  mặc định +1.9 pp (p = 0.055, sát ngưỡng).
- Bản 0.05 chạy từ bản sao có sửa 1 dòng (`~/qtool-exp/qtool-damp`), code repo **chưa** đổi. Checkpoint thí nghiệm:
  `rtx4070:~/qtool-exp/runs*/Qwen3.5-4B/checkpoints/INT4-W4A16`.
- Đề xuất (Nhân quyết): đổi mặc định GPTQ `dampening_frac` thành 0.05 (khớp recipe RedHat), có test, rồi chạy lại acceptance.

## Đính chính: GPTQ không tất định giữa các lần chạy (2026-11-01 tối)

Kết luận "dampening 0.05 đóng khoảng cách" ở trên **sai**: dựa trên 1 lần quantize. Chạy lại acceptance bằng code repo
(`9e370dd`, cùng calib cache sha256, cùng recipe.yaml, cùng version) ra checkpoint khác: 128/979 tensor lệch, mã int4 đổi
0.08% ở layer 0 → 15.7% ở layer 27 (sai số GPU nhỏ ở Hessian lật phép làm tròn, lan qua quantize tuần tự từng layer).
Eval vẫn tất định; nhiễu nằm ở bước quantize. Mỗi lần quantize là một lần rút thăm.

Tất cả lần quantize INT4-W4A16 (gsm8k full, strict-match; RedHat 0.8105):

| calib | dampening | strict | vs RedHat (McNemar) |
|---|---|---|---|
| ultrachat 512×2048, chat | 0.01 | 0.7870 | 99/68, p = 0.020 |
| Open-Platypus 1024×8192, text | 0.01 | 0.7900 | 89/62, p = 0.034 |
| ultrachat 512×2048, chat | 0.05 | 0.8059 | 84/78, p = 0.69 |
| ultrachat 512×2048, chat | 0.05 | 0.7688 | 112/57, p < 0.001 |
| Open-Platypus 1024×8192, text (recipe RedHat) | 0.05 | 0.7794 | 86/45, p < 0.001 |
| Open-Platypus 1024×8192, text (recipe RedHat) | 0.05 | 0.7953 | 79/59, p = 0.105 |
| Open-Platypus 1024×8192, text (recipe RedHat) | 0.05 | 0.7718 | 98/47, p < 0.001 |

- Recipe RedHat ×3: trung bình 0.7822 (−2.8 pp), dải 0.772–0.795; cả 3 thấp hơn RedHat. 7 lần tổng: chỉ 1 lần chạm RedHat.
- Dao động giữa các lần (~±1.5 pp) lớn hơn stderr eval (1.1 pp): tiêu chí "|Δ| ≤ stderr" trên 1 lần quantize không dùng được.
- Chưa có bằng chứng 0.05 tốt hơn 0.01 (5 lần 0.05 trung bình 0.7842 vs 2 lần 0.01 trung bình 0.7885).
- Config checkpoint so với RedHat: trùng (group 128, actorder static, symmetric, cùng tập ignore, cùng dampening, cùng
  calib Platypus 1024 shuffle seed 42). Khác còn lại: llmcompressor 0.13.0 / compressed-tensors 0.18.0 (qtool) vs
  compressed-tensors 0.14.1.a20260325 (RedHat, bản nightly cũ hơn); `add_special_tokens` True (qtool, mode text) vs False;
  thứ tự mẫu calib; GPU. RedHat có thể cũng là một lần rút thăm may (không biết họ chạy mấy lần).
- Checkpoint: `rtx4070:~/qtool-exp/rh{1,2,3}`; raw INT4 0.01 cũ: `~/qtool-exp/raw-INT4-W4A16-damp001`.

## Script RedHat chạy trong image qtool (2026-11-02)

Script tạo model trên model card RedHat, chạy nguyên văn trong `qtool-compress:cu129-4d4b1be4056f` (llmcompressor 0.13.0);
sửa duy nhất: `sequential_targets` chuyển từ `GPTQModifier` sang `oneshot()` (0.13 báo `extra_forbidden`, tức RedHat dùng
API cũ hơn). Eval bằng `qtool eval`, so cặp với RedHat (strict-match 0.8105).

| Lần | strict | vs RedHat (McNemar) |
|---|---|---|
| script RedHat #1 | 0.8089 | 66/68, p = 0.93 |
| script RedHat #2 | 0.7953 | 61/81, p = 0.11 |

- Dữ liệu calib qtool (`--calib platypus1024.jsonl --calib-samples 1024 --calib-seq-len 8192`) và của script RedHat
  **trùng từng token, cùng thứ tự** (1024 mẫu, 76 976 token); `add_special_tokens` không thêm token nào với tokenizer Qwen3.5.
  Recipe, ignore, dampening cũng trùng. Khác còn lại không ảnh hưởng số học: qtool truyền `processor=tok` và `dtype` tường minh.
- 5 lần cùng input (qtool ×3 + script ×2): 0.7718, 0.7794, 0.7953, 0.7953, 0.8089, trung bình 0.790. RedHat 0.8105 nằm ở
  đỉnh dải và script #1 đạt lại được trên llmcompressor 0.13: **không cần giả thuyết version thư viện**.
- Kết luận: qtool không có lỗi chất lượng; khoảng cách là dao động GPTQ giữa các lần (~±2 pp gsm8k), checkpoint RedHat là
  một lần rút thăm tốt. Acceptance cần tiêu chí tính đến dao động này.
