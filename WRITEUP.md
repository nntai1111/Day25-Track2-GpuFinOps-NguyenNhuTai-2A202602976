# Lab 25 — GPU FinOps: Write-up

**Họ tên:** Nguyễn Như Tài · **MSSV:** 2A202602976 · **Track 2 · Day 25**

Kết quả kiểm tra: `python verify.py` → **11/11 checks passed** · `pytest -q` → **24 passed** (15 test gốc, không sửa, cộng 9 test tôi viết cho phần mở rộng trong `tests/test_extensions.py`). Mọi số liệu dưới đây lấy từ `python missions/run_all.py` (seed = 25) và khớp với `outputs/report.md`.

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Tổng chi phí / tháng | **$27,133** | **$15,276** | **$11,857 (44%)** |
| Inference $/1M-token | $6.488 | $1.272 | 5.1× rẻ hơn (-80.4%) |
| GPU job (training/serving) / tháng | $25,667 on-demand | $15,877 | -38.1% |

Baseline: mọi request chạy model large, không cache, không batch; mọi GPU job thuê on-demand.

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | $/tháng | % tổng |
|---|---|---|
| Purchasing (spot/reserved) | $9,790 | 83% |
| Inference (cascade $1,122 · batch $54 · cache $2) | $1,179 | 10% |
| Kill idle GPUs | $600 | 5% |
| Right-size theo MBU | $288 | 2% |

**Purchasing đóng góp nhiều nhất**, vì 95% hóa đơn của NimbusAI là GPU-giờ ($25.7k) chứ không phải token ($1.5k). Trong inference, **cascade chiếm gần hết**: 79% request đã route về model small, mà small rẻ hơn large 15× (input) đến 37× (output). Cache và batch chỉ còn tác động lên phần đã rẻ sau cascade, vì các discount *nhân* với nhau chứ không cộng. Vì vậy, khi đo theo $/1M-token, inference là đòn bẩy mạnh nhất (5.1×). Khi đo theo tổng $, purchasing mới là đòn bẩy lớn nhất. Hai cách đo cho kết quả ngược nhau, nên báo cáo cần trình bày cả hai.

## 3. GPU-Util Lie

`gpu-h100-4` có util **98.2%** nhưng **MFU 0.194, MBU 0.207**. `gpu-a10g-1` có util 96.9%, MFU 0.268. `nvidia-smi` chỉ đo tỷ lệ thời gian *có kernel đang chạy*, không đo Tensor Core hay băng thông HBM thực sự được dùng bao nhiêu. Ở `gpu-h100-4`, cả compute lẫn băng thông đều thấp, tức GPU đang *chờ*: data loader chậm, batch nhỏ, đồng bộ giữa các GPU, hoặc overhead khi launch kernel. Các H100 cùng cụm đạt MFU ~0.41, nên cùng một khối lượng công việc trên `gpu-h100-4` tốn khoảng **2.1× tiền**. Nếu so với MFU tham chiếu 0.40, cả hai GPU "lie" đốt **~$1,165/tháng** cho FLOPs không nhận được. Right-sizing không giải quyết được `gpu-h100-4`: peak 225 TFLOPs và 67 GB VRAM thì không GPU NVIDIA nào rẻ hơn đáp ứng được. Cách sửa phải nằm ở phần mềm: profile bằng DCGM/Nsight, tăng prefetch, tăng micro-batch, dùng CUDA Graphs.

## 4. Phần mở rộng đã làm (5/5)

**Ext 1 — `recommend_tier()` v2** (`finops/pricing.py`, M3). Thay luật cứng bằng cách tính **$/giờ-GPU thực dùng** cho mọi lựa chọn hợp lệ, gồm:
- interruption rate riêng cho từng loại GPU (H100 3%/h, A10G 10%/h);
- reservation phải trả cả giờ idle (24/giờ-dùng-mỗi-ngày) và cả những tháng còn lại sau khi workload đã bị bỏ (term/lifetime);
- job dưới 28 ngày không được cam kết reserved.

Savings đổi từ **39.1% xuống 38.1%**, vì policy cũ chỉ tính reserved cho giờ thực dùng: L4 chạy 18h/ngày thực tế tốn $1,296, không phải $972. Insight: **3yr chỉ thắng 1yr khi workload sống quá 36·r3/r1 ≈ 25–27 tháng**. Với H100, break-even thực của reserved 1yr là 80% duty (vì chỉ giảm 20%), không phải 55%. M3 in thêm ma trận GPU × duty × interruptible × lifetime.

**Ext 2 — Right-sizing theo MBU** (M1). In bảng $/GB-VRAM và $/(TB/s) của 7 loại GPU. Với mỗi GPU có MBU < 0.35, chọn GPU rẻ nhất đủ đáp ứng peak VRAM (+10%), peak BW (≤ 80% peak) và peak FLOPs (≤ 60%). Kết quả: 2× A10G → L4, tiết kiệm **$288/tháng**. Mapping cũ (H100 → A100) là không khả thi. Đây cũng là câu trả lời cho việc vì sao không chọn GPU rẻ nhất theo $/giờ: L4 rẻ nhất nhưng chỉ có 24 GB và 0.3 TB/s, đắt nhất theo $/TB/s (2.67), nên decode sẽ chậm hẳn. Phương án MI300X (ROCm) tiết kiệm $396/tháng/GPU nhưng cần port phần mềm.

**Ext 3 — `cache_is_worth_it()`**. Cache có lợi khi `reads × giá_input × 0.9 > chi phí ghi`. Với kiểu Anthropic (ghi 1.25×, TTL 5 phút), break-even là **0.28 lượt đọc/lần ghi** cho cả hai tier. Với kiểu Gemini (phí lưu $1/1M/h, TTL 60 phút), small cần **5.56 lượt** còn large chỉ cần 0.37. Đo trên dữ liệu thật theo cửa sổ TTL: prefix large chỉ có 0.05–0.24 lượt đọc/5 phút nên **tắt cache**, prefix small có 0.63–1.26 lượt nên **bật cache**. Áp vào M2, optimized tăng từ $8.48 lên **$9.58/ngày**, nghĩa là báo cáo gốc đã tính dư ~$33/tháng tiền tiết kiệm từ cache.

**Ext 4 — Ngân sách reasoning**. Reasoning chiếm **8.4% request, 16.5% chi phí $, nhưng 94% năng lượng**. Mỗi request reasoning tốn 2.2× $ nhưng **172.6× Wh**, vì hệ số năng lượng 80× (chuỗi decode dài, tuần tự, bị giới hạn bởi băng thông bộ nhớ, KV-cache phình to) nhân thêm khoảng 2× token do thinking tokens. Toàn bộ reasoning nằm ở team eval (48% traffic của team). Quy tắc đề xuất: mỗi team chỉ được dùng reasoning cho tối đa **10% request**, ưu tiên context dài nhất (eval: input ≥ 3,382 token). Kết quả: 160 request bị hạ cấp, tiết kiệm **$0.85/ngày (10%) và 22,690 Wh/ngày (72%)**.

**Ext 5 — Carbon-aware scheduling** (`missions/m6_carbon_scheduling.py`). 5 job interruptible dùng 5,072 kWh/tháng (PUE 1.2). Chuyển từ us-east-1 sang **europe-north1** (sạch nhất) giảm **1,775 kgCO2e/tháng (-92%)** và $152 tiền điện. Sang **us-east-wa** (rẻ nhất và cân bằng nhất) giảm 76% CO2 và **$330/tháng**. Vùng "tối ưu" tùy mục tiêu: chọn ESG thì europe-north1, chọn $ thì us-east-wa. Job batch không nhạy latency nên chuyển được, còn inference real-time (RTT 110 ms từ Mỹ sang Na Uy) nên ở lại us-east.

**Insight quan trọng nhất:** nhiều con số "tiết kiệm" trong bản gốc là lạc quan, vì bỏ qua giờ idle của reservation, chi phí ghi cache và các ràng buộc VRAM/FLOPs. Khi tính đúng, tổng tiết kiệm giảm từ 46% xuống 44% nhưng là con số có thể cam kết với CFO.

## 5. Khuyến nghị cho NimbusAI — 3 hành động đầu tiên

1. **Tuần 1: auto-stop GPU idle và mua spot cho job có checkpoint.** Tắt instance khi util < 10% trong 30 phút (+$600/tháng). Chuyển 5 job interruptible sang spot, chạy ở us-east-wa (-$5,924/tháng tiền GPU, thêm -$330 tiền điện và -76% CO2). Cả hai không cần thay đổi code model.
2. **Tháng 1: cam kết reserved 3yr chỉ cho 3 service inference 24/7.** Đây là các service có vòng đời ≥ 26 tháng (chat, rag, search). Trước khi ký, sửa util-lie trên `gpu-h100-4` bằng profiling để không khóa 3 năm vào một cụm đang chạy ở 1/2 hiệu suất.
3. **Liên tục: quản trị theo $/1M-token và Wh/request ở gateway.** Giữ cascade (≥ 79% sang small), bật cache chỉ cho prefix vượt break-even, giới hạn reasoning 10%/team, và chargeback theo team (tag coverage 92% > ngưỡng 80%) để từng team tự chịu trách nhiệm chi phí.
