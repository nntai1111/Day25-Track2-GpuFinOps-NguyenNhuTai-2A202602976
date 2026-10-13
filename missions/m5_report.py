"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

All analysis sections are generated from the mission results, so every number in
report.md matches what the missions print.

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m6_carbon_scheduling

DAYS = 30
EFFORT = {
    "Purchasing (spot/reserved)": "Thấp–TB: cam kết reserved cho 3 service 24/7 + chuyển job train có checkpoint sang spot",
    "Inference (cascade/cache/batch)": "Thấp: routing/cache/batch là thay đổi config ở gateway",
    "Kill idle GPUs": "Rất thấp: tắt instance khi job xong (auto-stop theo DCGM idle)",
    "Right-size (MBU-based)": "TB: migrate A10G -> L4, cần test latency",
}
HEALTHY_MFU = 0.40  # good training MFU is ~0.35-0.45 (finops.metrics); the H100 peers here run ~0.41


def _lie_section(r1, cat) -> tuple[str, float]:
    rows, phantom = [], 0.0
    for lie in r1["lies"]:
        monthly = num(cat[lie["gpu_type"]]["on_demand_hr"]) * 24 * DAYS
        lost = max(0.0, 1 - lie["mfu"] / HEALTHY_MFU)
        phantom += monthly * lost
        rows.append(f"| {lie['gpu_id']} | {lie['gpu_type']} | {lie['gpu_util_pct']}% | {lie['mfu']} | {lie['mbu']} "
                    f"| ${monthly:,.0f} | {lost:.0%} | ${monthly * lost:,.0f} |")
    text = "\n".join([
        "## GPU-Util lie — vì sao 98% util vẫn là lãng phí",
        "",
        "| GPU | Loại | GPU-Util | MFU | MBU | Chi phí/tháng | Compute mất so với MFU 0.40 | Tiền trả cho FLOPs không nhận |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"**Tổng: ~${phantom:,.0f}/tháng** trả cho năng lực tính toán không được dùng.",
        "",
        "**Cơ chế:** `GPU-Util` của `nvidia-smi` chỉ đo *tỷ lệ thời gian có ít nhất một kernel đang chạy*, "
        "không đo bao nhiêu SM/Tensor Core thực sự làm việc. Một kernel nhỏ, một vòng chờ data loader, "
        "hay chuỗi kernel launch ngắn liên tục (launch overhead, không dùng CUDA Graphs) vẫn làm util = 100% "
        "trong khi Tensor Core gần như nghỉ. `gpu-h100-4` có util 98% nhưng MFU 0.19 và MBU 0.21 — cả compute "
        "lẫn băng thông HBM đều thấp, nghĩa là GPU *đang chờ* (input pipeline / đồng bộ giữa các GPU / batch quá nhỏ) "
        "chứ không bị giới hạn bởi phần cứng. Vì ta trả theo GPU-giờ, mọi giờ chạy ở MFU 0.19 thay vì ~0.41 như "
        "các H100 cùng cụm nghĩa là cùng một khối lượng công việc tốn **~2.1× tiền**.",
        "",
        "**Hành động:** right-sizing *không* giải quyết được `gpu-h100-4` (peak 225 TFLOPs + 67 GB VRAM — không GPU NVIDIA "
        "nào rẻ hơn đáp ứng được, xem Ext 2). Cách sửa là phần mềm: profile bằng Nsight/DCGM "
        "(`tensor_active`, `dram_active`), tăng `num_workers`/prefetch cho data loader, tăng micro-batch, bật CUDA Graphs.",
    ])
    return text, phantom


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r6 = m6_carbon_scheduling.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]
    idle_savings = r1["idle_waste_daily"] * DAYS
    # Ext 2: right-size only where a cheaper GPU fits VRAM + bandwidth + FLOPs (M1)
    rightsize_savings = r1["rightsize_monthly"]

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size (MBU-based)": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }
    # the template line says "Cheapest+cleanest" — they are different regions here, so name both
    sust["best_region"] = (f"{sust['best_region']} (cleanest, {sustainability.REGION_CARBON[sust['best_region']]} gCO2/kWh); "
                           f"{r6['cheapest']} is cheapest (${sustainability.REGION_PRICE_KWH[r6['cheapest']]}/kWh) "
                           f"and best balanced")

    # --- analysis sections ---
    rb = r2["reasoning"]
    inf_base_m, inf_opt_m = r2["baseline_daily"] * DAYS, r2["optimized_daily"] * DAYS
    unit = "\n".join([
        "## Unit economics — $/1M-token",
        "",
        "| | Baseline | Optimized | Thay đổi |",
        "|---|---|---|---|",
        f"| Inference $/ngày | ${r2['baseline_daily']:,.2f} | ${r2['optimized_daily']:,.2f} | -{r2['savings_pct']}% |",
        f"| Inference $/tháng | ${inf_base_m:,.0f} | ${inf_opt_m:,.0f} | -${inf_base_m - inf_opt_m:,.0f} |",
        f"| **$/1M-token** | **${r2['baseline_per_m']:.3f}** | **${r2['optimized_per_m']:.3f}** | "
        f"**{r2['baseline_per_m'] / r2['optimized_per_m']:.1f}× rẻ hơn** |",
        *[f"| ↳ đòn bẩy {k} | | -${v * DAYS:,.0f}/tháng | {v / r2['baseline_daily'] * 100:.1f}% baseline inference |"
          for k, v in r2["lever_daily"].items()],
        f"| GPU training/serving $/tháng | ${r3['on_demand_monthly']:,.0f} (on-demand) | ${r3['optimized_monthly']:,.0f} | -{r3['savings_pct']}% |",
        "",
        f"Baseline = mọi request chạy model large, không cache, không batch; mọi GPU job on-demand. "
        f"Tổng {r2['total_tokens']:,} token/ngày. Cache được tính *có điều kiện* (Ext 3): nếu bỏ cổng "
        f"`cache_is_worth_it()` thì optimized là ${r2['ungated_optimized_daily']:.2f}/ngày — tức báo cáo cũ "
        f"đã đếm cả tiết kiệm cache ở những prefix không đủ lượt đọc để bù chi phí ghi.",
    ])
    lever_rows = sorted(levers.items(), key=lambda kv: -kv[1])
    priorities = "\n".join([
        "## Phân tích đòn bẩy & thứ tự ưu tiên",
        "",
        "| # | Đòn bẩy | $/tháng | % tổng tiết kiệm | Công sức / rủi ro |",
        "|---|---|---|---|---|",
        *[f"| {i} | {k} | ${v:,.0f} | {v / sum(levers.values()) * 100:.0f}% | {EFFORT[k]} |"
          for i, (k, v) in enumerate(lever_rows, 1)],
        "",
        "**Thứ tự hành động theo ROI (tiết kiệm / công sức):**",
        "",
        f"1. **Kill idle GPUs ngay hôm nay** — ${levers['Kill idle GPUs']:,}/tháng, gần như không tốn công: "
        f"`gpu-h100-5` idle {next(s['idle_hours'] for s in r1['summary'] if s['gpu_id'] == 'gpu-h100-5')}h/ngày. Đặt policy auto-stop khi util < 10% trong 30 phút.",
        f"2. **Purchasing** — đòn bẩy lớn nhất (${levers['Purchasing (spot/reserved)']:,}/tháng) vì phần lớn hóa đơn là GPU-giờ "
        f"chứ không phải token. Spot cho 5 job interruptible (đã có checkpoint), reserved 3yr chỉ cho service chạy 24/7 "
        f"có vòng đời ≥ ~26 tháng (Ext 1).",
        f"3. **Inference levers** — ${levers['Inference (cascade/cache/batch)']:,}/tháng tuyệt đối nhỏ nhưng là "
        f"{r2['savings_pct']}% của phần inference và giảm $/1M-token {r2['baseline_per_m'] / r2['optimized_per_m']:.1f}×; "
        f"cascade đóng góp nhiều nhất (${r2['lever_daily']['cascade'] * DAYS:,.0f}/tháng) vì giá model small rẻ hơn 15–37× "
        f"và {r2['small_share']:.0%} request đã route về small; cache (${r2['lever_daily']['cache'] * DAYS:,.0f}) và batch "
        f"(${r2['lever_daily']['batch'] * DAYS:,.0f}) chỉ còn tác động lên phần đã rẻ sau cascade — discount nhân chứ không cộng.",
        f"4. **Sửa util-lie bằng phần mềm** (~${_lie_section(r1, cat)[1]:,.0f}/tháng compute bị bỏ phí) rồi mới right-size A10G -> L4 "
        f"(${levers['Right-size (MBU-based)']:,}/tháng).",
        f"5. **Ngân sách reasoning** — cap 10% traffic mỗi team tiết kiệm ${rb['usd_saved_daily'] * DAYS:,.0f}/tháng nhưng "
        f"giảm {rb['wh_saved_daily'] / rb['total_wh'] * 100:.0f}% năng lượng inference (Ext 4).",
    ])
    lie_text, _ = _lie_section(r1, cat)
    best_reg = r6["regions"]
    sust_text = "\n".join([
        "## Sustainability — carbon gắn với chi phí",
        "",
        f"- Query thường (~{median_tokens} token): {wh:.2f} Wh, {sust['carbon_g']:.3f} gCO2e ở us-east-1. "
        f"Query reasoning tốn **{rb['wh_per_req_ratio']}× Wh** (80× hệ số năng lượng × ~2× token do thinking tokens).",
        f"- Reasoning chỉ chiếm {rb['reasoning_traffic_pct']}% request và {rb['reasoning_cost_pct']}% chi phí $, "
        f"nhưng **{rb['reasoning_energy_pct']}% năng lượng** inference — $ không phản ánh carbon, cần đo Wh riêng.",
        f"- Job interruptible tiêu thụ {r6['total_kwh']:,} kWh/tháng (PUE 1.2). So sánh vùng (Ext 5):",
        "",
        "| Vùng | $/kWh | gCO2/kWh | Tiền điện/tháng | kgCO2e/tháng | RTT từ user (ms) |",
        "|---|---|---|---|---|---|",
        *[f"| {r['region']} | {r['usd_kwh']} | {r['gco2_kwh']} | ${r['energy_usd']:,.0f} | {r['carbon_kg']:,.0f} | {r['rtt_ms']} |"
          for r in sorted(best_reg, key=lambda r: r["score"])],
        "",
        f"- **Sạch nhất:** {r6['cleanest']} (-{r6['cleanest_carbon_saved_pct']}% CO2, -${r6['cleanest_usd_saved']}/tháng). "
        f"**Rẻ nhất & cân bằng nhất:** {r6['balanced']} (-{r6['balanced_carbon_saved_pct']}% CO2, -${r6['balanced_usd_saved']}/tháng).",
        f"- Carbon và chi phí đi cùng chiều: vùng bẩn nhất (europe-central2, 660 g/kWh) cũng đắt nhất ($0.18/kWh); "
        f"us-east-1 hiện tại vừa đắt gấp 2.2× vừa bẩn gấp 4.2× us-east-wa.",
        f"- Khuyến nghị: chuyển job train/eval interruptible sang **{r6['cleanest']}** nếu công ty ưu tiên ESG "
        f"(job batch không nhạy latency, chỉ cần lưu ý data residency/egress), hoặc **{r6['balanced']}** nếu ưu tiên $; "
        f"giữ inference real-time gần user (us-east).",
    ])
    ext_text = "\n".join([
        "## Phần mở rộng đã làm (Your Turn)",
        "",
        "| Ext | Thay đổi | Trước | Sau |",
        "|---|---|---|---|",
        f"| 1. `recommend_tier()` v2 | Interruption rate theo GPU + 1yr vs 3yr + tính giờ idle của reservation | "
        f"{r3['v1_savings_pct']}% (v1) | {r3['savings_pct']}% (v2, trung thực hơn) |",
        f"| 2. Right-sizing theo MBU | Chọn GPU rẻ nhất vừa VRAM+BW+FLOPs (dùng $/GB-VRAM, peak_bw_tbs) | "
        f"map 1 tier xuống (gồm H100->A100 không khả thi) | ${r1['rightsize_monthly']:,.0f}/tháng (2× A10G->L4) |",
        f"| 3. `cache_is_worth_it()` | Chỉ tính cache khi reads/write > break-even (0.28 lượt với 5-min TTL) | "
        f"${r2['ungated_optimized_daily']:.2f}/ngày | ${r2['optimized_daily']:.2f}/ngày |",
        f"| 4. Ngân sách reasoning | Cap reasoning 10%/team, ưu tiên context dài (giá chưa gate cache) | "
        f"${rb['total_usd']:.2f}/ngày, {rb['total_wh']:,.0f} Wh | ${rb['capped_usd']:.2f}/ngày, {rb['capped_wh']:,.0f} Wh |",
        f"| 5. Carbon-aware scheduling | Dời job interruptible khỏi us-east-1 | "
        f"{r6['home_carbon_kg']:,} kgCO2e/tháng | {min(r['carbon_kg'] for r in r6['regions']):,.0f} kgCO2e ({r6['cleanest']}) |",
    ])

    md = report.build_report(baseline, optimized, levers, sustainability=sust,
                             extra_sections=[unit, priorities, lie_text, sust_text, ext_text])
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"), baseline=baseline)

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
