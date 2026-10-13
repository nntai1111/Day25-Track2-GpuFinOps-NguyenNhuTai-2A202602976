"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Extension 2: MBU-based right-sizing — for GPUs with low bandwidth utilization,
find the cheapest GPU that still fits the workload's VRAM, bandwidth and FLOPs.

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30
LOW_MBU = 0.35          # below this the GPU's HBM bandwidth is mostly wasted
TARGET_MBU = 0.80       # max sustained BW we plan for on the new GPU (headroom for peaks)
TARGET_MFU = 0.60       # max sustained FLOPs we plan for on the new GPU
VRAM_HEADROOM = 1.10    # +10% VRAM over the observed peak (KV-cache growth, fragmentation)
CROSS_VENDOR = {"MI300X"}  # needs a ROCm port — reported, but not in the headline number


def gpu_unit_prices(cat: dict) -> list[dict]:
    """$/GB-VRAM-hr and $/(TB/s)-hr per GPU type: what you pay for memory capacity and speed."""
    out = []
    for g, c in cat.items():
        od, hbm, bw = num(c["on_demand_hr"]), num(c["hbm_gb"]), num(c["peak_bw_tbs"])
        out.append({"gpu_type": g, "on_demand_hr": od, "hbm_gb": hbm, "peak_bw_tbs": bw,
                    "usd_per_gb_vram": round(od / hbm, 4), "usd_per_tbs": round(od / bw, 3)})
    return sorted(out, key=lambda r: r["usd_per_gb_vram"])


def rightsize(profile: dict, cat: dict, allow_cross_vendor: bool = False):
    """Cheapest GPU type whose VRAM, bandwidth and FLOPs cover the observed peak demand.

    profile: gpu_type, peak_mem_gb, peak_bw_tbs, peak_tflops (observed maxima over 24h).
    Returns (gpu_type, reason) — the current type when nothing cheaper fits.
    """
    cur = profile["gpu_type"]
    best, best_hr = cur, num(cat[cur]["on_demand_hr"])
    for g, c in cat.items():
        if g in CROSS_VENDOR and not allow_cross_vendor:
            continue
        fits = (num(c["hbm_gb"]) >= profile["peak_mem_gb"] * VRAM_HEADROOM
                and num(c["peak_bw_tbs"]) * TARGET_MBU >= profile["peak_bw_tbs"]
                and num(c["peak_tflops_fp16"]) * TARGET_MFU >= profile["peak_tflops"])
        if fits and num(c["on_demand_hr"]) < best_hr:
            best, best_hr = g, num(c["on_demand_hr"])
    if best == cur:
        return cur, "no cheaper GPU fits VRAM+BW+FLOPs peak -> fix software / consolidate"
    c = cat[best]
    return best, (f"{num(c['hbm_gb']):.0f}GB >= {profile['peak_mem_gb'] * VRAM_HEADROOM:.0f}GB, "
                  f"{num(c['peak_bw_tbs'])}TB/s x{TARGET_MBU} >= {profile['peak_bw_tbs']:.2f}TB/s")


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0,
                               "mem": [], "bw": [], "tflops": []})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        a["mem"].append(num(r["mem_used_gb"]))
        a["bw"].append(num(r["achieved_bw_tbs"]))
        a["tflops"].append(num(r["achieved_tflops"]))
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "peak_mem_gb": max(a["mem"]), "peak_bw_tbs": max(a["bw"]), "peak_tflops": max(a["tflops"]),
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: MBU right-sizing ---
    rightsizing = []
    for s in sorted(summary, key=lambda x: x["mbu"]):
        if s["mbu"] >= LOW_MBU:
            continue
        cur_hr = num(cat[s["gpu_type"]]["on_demand_hr"])
        tgt, why = rightsize(s, cat)
        alt, _ = rightsize(s, cat, allow_cross_vendor=True)
        tgt_hr = num(cat[tgt]["on_demand_hr"])
        rightsizing.append({
            "gpu_id": s["gpu_id"], "gpu_type": s["gpu_type"], "mbu": s["mbu"], "target": tgt,
            "reason": why, "cross_vendor_option": alt if alt != tgt else "",
            "monthly_savings": round((cur_hr - tgt_hr) * 24 * DAYS, 2),
            "savings_pct": round((1 - tgt_hr / cur_hr) * 100, 1),
            "cross_vendor_monthly_savings": round((cur_hr - num(cat[alt]["on_demand_hr"])) * 24 * DAYS, 2),
        })
    rightsize_monthly = sum(r["monthly_savings"] for r in rightsizing)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- Ext 2: GPU unit prices (on-demand) --")
        print(f"{'type':8}{'$/hr':>7}{'HBM GB':>8}{'TB/s':>7}{'$/GB-VRAM':>11}{'$/TB/s':>9}")
        for u in gpu_unit_prices(cat):
            print(f"{u['gpu_type']:8}{u['on_demand_hr']:>7.2f}{u['hbm_gb']:>8.0f}{u['peak_bw_tbs']:>7.2f}"
                  f"{u['usd_per_gb_vram']:>11.4f}{u['usd_per_tbs']:>9.3f}")
        print(f"\n-- Ext 2: right-sizing GPUs with MBU < {LOW_MBU} --")
        print(f"{'GPU':13}{'type':6}{'MBU':>6}{'peakGB':>8}{'peakTB/s':>9}{'peakTF':>8}  {'->':4}{'$/mo saved':>11}  reason")
        by_id = {s['gpu_id']: s for s in summary}
        for r in rightsizing:
            s = by_id[r["gpu_id"]]
            alt = f"  [ROCm option: {r['cross_vendor_option']} ${r['cross_vendor_monthly_savings']:,.0f}/mo]" if r["cross_vendor_option"] else ""
            print(f"{r['gpu_id']:13}{r['gpu_type']:6}{r['mbu']:>6}{s['peak_mem_gb']:>8}{s['peak_bw_tbs']:>9}{s['peak_tflops']:>8}"
                  f"  {r['target']:6}${r['monthly_savings']:>9,.0f}  {r['reason']}{alt}")
        print(f"Right-sizing all low-MBU GPUs (same vendor): ${rightsize_monthly:,.0f}/month")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "rightsizing": rightsizing, "rightsize_monthly": round(rightsize_monthly, 2)}


if __name__ == "__main__":
    run()
