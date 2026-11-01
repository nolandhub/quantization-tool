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
