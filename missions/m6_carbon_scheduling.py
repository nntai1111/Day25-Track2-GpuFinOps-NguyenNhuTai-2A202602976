"""M6 (Extension 5) — Carbon-aware Scheduling for interruptible jobs (deck §11).

Interruptible (checkpointed) jobs are latency-insensitive, so they can follow the
cheapest / cleanest grid. Compare every region on electricity $ and gCO2e.

Run: python missions/m6_carbon_scheduling.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability as sus

DAYS = 30
PUE = 1.2              # facility overhead (cooling, power conversion) on top of GPU board power
HOME_REGION = "us-east-1"
# Rough RTT from NimbusAI's users (US East) — matters for serving, not for batch training.
RTT_MS = {"us-east-1": 5, "us-east-wa": 15, "us-west-2": 70, "europe-north1": 110, "europe-central2": 105}


def job_kwh(job: dict, cat: dict) -> float:
    gpu_hours = num(job["hours_per_day"]) * DAYS * int(num(job["num_gpus"]))
    return gpu_hours * num(cat[job["gpu_type"]]["watts"]) / 1000.0 * PUE


def region_table(kwh: float) -> list[dict]:
    rows = []
    for reg, g in sus.REGION_CARBON.items():
        wh = kwh * 1000
        rows.append({"region": reg, "usd_kwh": sus.REGION_PRICE_KWH[reg], "gco2_kwh": g,
                     "energy_usd": sus.energy_cost_usd(wh, reg), "carbon_kg": sus.carbon_g(wh, reg) / 1000,
                     "rtt_ms": RTT_MS.get(reg, 0)})
    # balanced: equal weight on min-max normalised $ and CO2
    lo_u, hi_u = min(r["usd_kwh"] for r in rows), max(r["usd_kwh"] for r in rows)
    lo_c, hi_c = min(r["gco2_kwh"] for r in rows), max(r["gco2_kwh"] for r in rows)
    for r in rows:
        r["score"] = 0.5 * (r["usd_kwh"] - lo_u) / (hi_u - lo_u) + 0.5 * (r["gco2_kwh"] - lo_c) / (hi_c - lo_c)
    return rows


def run(verbose: bool = True) -> dict:
    cat = catalog_by_type()
    jobs = [j for j in load_csv("workloads.csv") if int(num(j["interruptible"]))]
    per_job = []
    for j in jobs:
        kwh = job_kwh(j, cat)
        rows = {r["region"]: r for r in region_table(kwh)}
        clean = min(rows.values(), key=lambda r: r["gco2_kwh"])
        per_job.append({"job_id": j["job_id"], "kwh": kwh,
                        "home_kg": rows[HOME_REGION]["carbon_kg"], "clean_kg": clean["carbon_kg"],
                        "home_usd": rows[HOME_REGION]["energy_usd"], "clean_usd": clean["energy_usd"]})
    total_kwh = sum(p["kwh"] for p in per_job)
    table = region_table(total_kwh)
    by = {r["region"]: r for r in table}
    cheapest = min(table, key=lambda r: r["usd_kwh"])["region"]
    cleanest = min(table, key=lambda r: r["gco2_kwh"])["region"]
    balanced = min(table, key=lambda r: r["score"])["region"]
    home = by[HOME_REGION]
    result = {
        "total_kwh": round(total_kwh), "home_region": HOME_REGION,
        "cheapest": cheapest, "cleanest": cleanest, "balanced": balanced,
        "home_carbon_kg": round(home["carbon_kg"]), "home_energy_usd": round(home["energy_usd"]),
        "regions": table, "jobs": per_job,
    }
    for k, reg in (("cleanest", cleanest), ("balanced", balanced), ("cheapest", cheapest)):
        result[f"{k}_carbon_saved_kg"] = round(home["carbon_kg"] - by[reg]["carbon_kg"])
        result[f"{k}_carbon_saved_pct"] = round((1 - by[reg]["carbon_kg"] / home["carbon_kg"]) * 100, 1)
        result[f"{k}_usd_saved"] = round(home["energy_usd"] - by[reg]["energy_usd"])

    if verbose:
        print("== M6 Carbon-aware Scheduling (Ext 5) ==")
        print(f"interruptible jobs: {len(jobs)}  energy {total_kwh:,.0f} kWh/month (PUE {PUE})")
        print(f"{'job':18}{'kWh':>8}{'kgCO2 ' + HOME_REGION:>20}{'kgCO2 ' + cleanest:>22}{'saved':>8}")
        for p in per_job:
            print(f"{p['job_id']:18}{p['kwh']:>8,.0f}{p['home_kg']:>20,.0f}{p['clean_kg']:>22,.0f}"
                  f"{(1 - p['clean_kg'] / p['home_kg']) * 100:>7.0f}%")
        print(f"\n{'region':17}{'$/kWh':>7}{'gCO2/kWh':>10}{'energy $':>10}{'kgCO2e':>9}{'RTT ms':>8}{'score':>7}")
        for r in sorted(table, key=lambda r: r["score"]):
            print(f"{r['region']:17}{r['usd_kwh']:>7.3f}{r['gco2_kwh']:>10}{r['energy_usd']:>10,.0f}"
                  f"{r['carbon_kg']:>9,.0f}{r['rtt_ms']:>8}{r['score']:>7.2f}")
        print(f"\ncheapest $: {cheapest} | cleanest CO2: {cleanest} | balanced: {balanced}")
        for k in ("cleanest", "balanced", "cheapest"):
            print(f"move all interruptible jobs {HOME_REGION} -> {result[k]:15}: "
                  f"-{result[k + '_carbon_saved_kg']:,} kgCO2e/month ({result[k + '_carbon_saved_pct']}%), "
                  f"-${result[k + '_usd_saved']:,}/month electricity")
    return result


if __name__ == "__main__":
    run()
