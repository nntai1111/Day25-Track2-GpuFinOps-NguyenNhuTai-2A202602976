# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $15,276  
**Projected savings:** $11,857  (**44%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,179 |
| Purchasing (spot/reserved) | $9,790 |
| Right-size (MBU-based) | $288 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1 (cleanest, 30 gCO2/kWh); us-east-wa is cheapest ($0.055/kWh) and best balanced

## Unit economics — $/1M-token

| | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Inference $/ngày | $48.87 | $9.58 | -80.4% |
| Inference $/tháng | $1,466 | $287 | -$1,179 |
| **$/1M-token** | **$6.488** | **$1.272** | **5.1× rẻ hơn** |
| ↳ đòn bẩy cascade | | -$1,122/tháng | 76.5% baseline inference |
| ↳ đòn bẩy cache | | -$2/tháng | 0.2% baseline inference |
| ↳ đòn bẩy batch | | -$54/tháng | 3.7% baseline inference |
| GPU training/serving $/tháng | $25,667 (on-demand) | $15,877 | -38.1% |

Baseline = mọi request chạy model large, không cache, không batch; mọi GPU job on-demand. Tổng 7,533,027 token/ngày. Cache được tính *có điều kiện* (Ext 3): nếu bỏ cổng `cache_is_worth_it()` thì optimized là $8.48/ngày — tức báo cáo cũ đã đếm cả tiết kiệm cache ở những prefix không đủ lượt đọc để bù chi phí ghi.

## Phân tích đòn bẩy & thứ tự ưu tiên

| # | Đòn bẩy | $/tháng | % tổng tiết kiệm | Công sức / rủi ro |
|---|---|---|---|---|
| 1 | Purchasing (spot/reserved) | $9,790 | 83% | Thấp–TB: cam kết reserved cho 3 service 24/7 + chuyển job train có checkpoint sang spot |
| 2 | Inference (cascade/cache/batch) | $1,179 | 10% | Thấp: routing/cache/batch là thay đổi config ở gateway |
| 3 | Kill idle GPUs | $600 | 5% | Rất thấp: tắt instance khi job xong (auto-stop theo DCGM idle) |
| 4 | Right-size (MBU-based) | $288 | 2% | TB: migrate A10G -> L4, cần test latency |

**Thứ tự hành động theo ROI (tiết kiệm / công sức):**

1. **Kill idle GPUs ngay hôm nay** — $600/tháng, gần như không tốn công: `gpu-h100-5` idle 8h/ngày. Đặt policy auto-stop khi util < 10% trong 30 phút.
2. **Purchasing** — đòn bẩy lớn nhất ($9,790/tháng) vì phần lớn hóa đơn là GPU-giờ chứ không phải token. Spot cho 5 job interruptible (đã có checkpoint), reserved 3yr chỉ cho service chạy 24/7 có vòng đời ≥ ~26 tháng (Ext 1).
3. **Inference levers** — $1,179/tháng tuyệt đối nhỏ nhưng là 80.4% của phần inference và giảm $/1M-token 5.1×; cascade đóng góp nhiều nhất ($1,122/tháng) vì giá model small rẻ hơn 15–37× và 79% request đã route về small; cache ($2) và batch ($54) chỉ còn tác động lên phần đã rẻ sau cascade — discount nhân chứ không cộng.
4. **Sửa util-lie bằng phần mềm** (~$1,165/tháng compute bị bỏ phí) rồi mới right-size A10G -> L4 ($288/tháng).
5. **Ngân sách reasoning** — cap 10% traffic mỗi team tiết kiệm $26/tháng nhưng giảm 72% năng lượng inference (Ext 4).

## GPU-Util lie — vì sao 98% util vẫn là lãng phí

| GPU | Loại | GPU-Util | MFU | MBU | Chi phí/tháng | Compute mất so với MFU 0.40 | Tiền trả cho FLOPs không nhận |
|---|---|---|---|---|---|---|---|
| gpu-h100-4 | H100 | 98.2% | 0.194 | 0.207 | $1,800 | 52% | $927 |
| gpu-a10g-1 | A10G | 96.9% | 0.268 | 0.302 | $720 | 33% | $238 |

**Tổng: ~$1,165/tháng** trả cho năng lực tính toán không được dùng.

**Cơ chế:** `GPU-Util` của `nvidia-smi` chỉ đo *tỷ lệ thời gian có ít nhất một kernel đang chạy*, không đo bao nhiêu SM/Tensor Core thực sự làm việc. Một kernel nhỏ, một vòng chờ data loader, hay chuỗi kernel launch ngắn liên tục (launch overhead, không dùng CUDA Graphs) vẫn làm util = 100% trong khi Tensor Core gần như nghỉ. `gpu-h100-4` có util 98% nhưng MFU 0.19 và MBU 0.21 — cả compute lẫn băng thông HBM đều thấp, nghĩa là GPU *đang chờ* (input pipeline / đồng bộ giữa các GPU / batch quá nhỏ) chứ không bị giới hạn bởi phần cứng. Vì ta trả theo GPU-giờ, mọi giờ chạy ở MFU 0.19 thay vì ~0.41 như các H100 cùng cụm nghĩa là cùng một khối lượng công việc tốn **~2.1× tiền**.

**Hành động:** right-sizing *không* giải quyết được `gpu-h100-4` (peak 225 TFLOPs + 67 GB VRAM — không GPU NVIDIA nào rẻ hơn đáp ứng được, xem Ext 2). Cách sửa là phần mềm: profile bằng Nsight/DCGM (`tensor_active`, `dram_active`), tăng `num_workers`/prefetch cho data loader, tăng micro-batch, bật CUDA Graphs.

## Sustainability — carbon gắn với chi phí

- Query thường (~800 token): 0.24 Wh, 0.091 gCO2e ở us-east-1. Query reasoning tốn **172.6× Wh** (80× hệ số năng lượng × ~2× token do thinking tokens).
- Reasoning chỉ chiếm 8.4% request và 16.5% chi phí $, nhưng **94.0% năng lượng** inference — $ không phản ánh carbon, cần đo Wh riêng.
- Job interruptible tiêu thụ 5,072 kWh/tháng (PUE 1.2). So sánh vùng (Ext 5):

| Vùng | $/kWh | gCO2/kWh | Tiền điện/tháng | kgCO2e/tháng | RTT từ user (ms) |
|---|---|---|---|---|---|
| us-east-wa | 0.055 | 90 | $279 | 457 | 15 |
| us-west-2 | 0.07 | 120 | $355 | 609 | 70 |
| europe-north1 | 0.09 | 30 | $457 | 152 | 110 |
| us-east-1 | 0.12 | 380 | $609 | 1,928 | 5 |
| europe-central2 | 0.18 | 660 | $913 | 3,348 | 105 |

- **Sạch nhất:** europe-north1 (-92.1% CO2, -$152/tháng). **Rẻ nhất & cân bằng nhất:** us-east-wa (-76.3% CO2, -$330/tháng).
- Carbon và chi phí đi cùng chiều: vùng bẩn nhất (europe-central2, 660 g/kWh) cũng đắt nhất ($0.18/kWh); us-east-1 hiện tại vừa đắt gấp 2.2× vừa bẩn gấp 4.2× us-east-wa.
- Khuyến nghị: chuyển job train/eval interruptible sang **europe-north1** nếu công ty ưu tiên ESG (job batch không nhạy latency, chỉ cần lưu ý data residency/egress), hoặc **us-east-wa** nếu ưu tiên $; giữ inference real-time gần user (us-east).

## Phần mở rộng đã làm (Your Turn)

| Ext | Thay đổi | Trước | Sau |
|---|---|---|---|
| 1. `recommend_tier()` v2 | Interruption rate theo GPU + 1yr vs 3yr + tính giờ idle của reservation | 39.1% (v1) | 38.1% (v2, trung thực hơn) |
| 2. Right-sizing theo MBU | Chọn GPU rẻ nhất vừa VRAM+BW+FLOPs (dùng $/GB-VRAM, peak_bw_tbs) | map 1 tier xuống (gồm H100->A100 không khả thi) | $288/tháng (2× A10G->L4) |
| 3. `cache_is_worth_it()` | Chỉ tính cache khi reads/write > break-even (0.28 lượt với 5-min TTL) | $8.48/ngày | $9.58/ngày |
| 4. Ngân sách reasoning | Cap reasoning 10%/team, ưu tiên context dài (giá chưa gate cache) | $8.48/ngày, 31,675 Wh | $7.64/ngày, 8,986 Wh |
| 5. Carbon-aware scheduling | Dời job interruptible khỏi us-east-1 | 1,928 kgCO2e/tháng | 152 kgCO2e (europe-north1) |

_Figures are June-2026 as-of snapshots; re-baseline before acting._