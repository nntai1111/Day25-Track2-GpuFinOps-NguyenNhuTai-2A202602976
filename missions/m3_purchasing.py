"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Extension 1: compares the documented v1 policy with a cost-based v2 policy
(per-GPU spot interruption rate, 1yr vs 3yr term, idle commitment hours).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30
# Expected remaining life of the workload, used to price stranded commitment months.
# Production inference services are long-lived; training / dev jobs churn with each model.
LIFETIME_MONTHS = {"infer": 36, "train": 12, "dev": 12}


def _v1_cost(gpu_hours, tier, c, od):
    """Original policy accounting: reserved billed only for used hours at the 3yr rate."""
    if tier == "spot":
        return pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
    if tier == "reserved":
        return gpu_hours * num(c["reserved_3yr_hr"])
    return gpu_hours * od


def tier_matrix(cat, gpu_types=("H100", "A100", "A10G", "L4"), duties=(3, 8, 14, 18, 24),
                lifetimes=(12, 36)):
    """GPU type x duty cycle (h/day) x interruptible x lifetime -> v2 recommended tier."""
    out = []
    for g in gpu_types:
        for intr in (True, False):
            for life in lifetimes:
                row = {"gpu_type": g, "interruptible": intr, "lifetime_m": life}
                for h in duties:
                    rec = pricing.recommend_purchase(h, intr, g, cat[g], job_days=30, lifetime_months=life)
                    row[h] = rec["tier"] + (f"-{rec['term']}" if rec["term"] else "")
                out.append(row)
    return out


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = v1_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # v1 (documented baseline policy) — kept for the before/after comparison
        tier_v1 = pricing.recommend_tier(hpd, interruptible)
        v1_cost = _v1_cost(gpu_hours, tier_v1, c, od)

        # v2 (Extension 1) — cost-based, GPU-aware, term-aware
        rec = pricing.recommend_purchase(hpd, interruptible, gtype, c, job_days=num(j["days"]),
                                         lifetime_months=LIFETIME_MONTHS.get(j["kind"], 12))
        opt_cost = gpu_hours * rec["effective_hr"]

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        v1_monthly += v1_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": rec["tier"], "term": rec["term"],
                     "tier_v1": tier_v1, "interrupt_rate": rec["interrupt_rate"],
                     "on_demand": round(on_demand_cost), "optimized_v1": round(v1_cost),
                     "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    v1_pct = (on_demand_monthly - v1_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'v1 tier':10}{'v2 tier':16}{'p(int)/h':>9}{'on-demand':>11}{'v1 cost':>10}{'v2 cost':>10}")
        for r in recs:
            v2 = r["tier"] + (f"-{r['term']}" if r["term"] else "")
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier_v1']:10}{v2:16}{r['interrupt_rate']:>9.0%}"
                  f"${r['on_demand']:>10,}${r['optimized_v1']:>9,}${r['optimized']:>9,}")
        print(f"\nmonthly on-demand ${on_demand_monthly:,.0f}")
        print(f"  v1 policy (docs)     -> ${v1_monthly:,.0f}  ({v1_pct:.1f}% saved)  [reserved billed for used hours only]")
        print(f"  v2 policy (Ext 1)    -> ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)  [idle + stranded commitment priced in]")

        print("\n1yr vs 3yr: workload lifetime (months) above which 3yr beats 1yr = 36 * r3 / r1")
        for g in ("H100", "A100", "A10G", "L4"):
            c = cat[g]
            print(f"  {g:6} r1=${num(c['reserved_1yr_hr']):.2f} r3=${num(c['reserved_3yr_hr']):.2f}"
                  f"  -> {pricing.term_break_even_months(num(c['reserved_1yr_hr']), num(c['reserved_3yr_hr'])):.1f} months")

        print("\nTier matrix (v2): GPU x interruptible x lifetime x hours/day")
        duties = (3, 8, 14, 18, 24)
        print(f"{'gpu':7}{'intr':6}{'life':6}" + "".join(f"{str(h)+'h':>16}" for h in duties))
        for row in tier_matrix(cat, duties=duties):
            print(f"{row['gpu_type']:7}{'yes' if row['interruptible'] else 'no':6}{str(row['lifetime_m'])+'m':6}"
                  + "".join(f"{row[h]:>16}" for h in duties))

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "v1_optimized_monthly": round(v1_monthly), "v1_savings_pct": round(v1_pct, 1)}


if __name__ == "__main__":
    run()
