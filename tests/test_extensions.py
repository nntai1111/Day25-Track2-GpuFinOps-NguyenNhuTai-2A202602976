"""Tests for the "Your Turn" extensions (added by the student; original tests untouched)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from finops import pricing
from missions._common import catalog_by_type
from missions import m1_efficiency_audit, m2_inference_levers, m6_carbon_scheduling

H100 = {"on_demand_hr": 2.5, "spot_hr": 1.5, "reserved_1yr_hr": 2.0, "reserved_3yr_hr": 1.4}


# --- Ext 1: recommend_tier v2 ---------------------------------------------------
def test_recommend_tier_legacy_api_unchanged():
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"


def test_reservation_pays_for_idle_and_stranded_hours():
    assert pricing.reserved_effective_hr(1.4, 24, 36, 36) == 1.4
    assert abs(pricing.reserved_effective_hr(1.4, 12, 36, 36) - 2.8) < 1e-9   # half idle -> 2x
    assert abs(pricing.reserved_effective_hr(1.4, 24, 36, 12) - 4.2) < 1e-9   # dies after 1yr of 3yr


def test_term_choice_depends_on_lifetime():
    assert abs(pricing.term_break_even_months(2.0, 1.4) - 25.2) < 1e-9
    long = pricing.recommend_purchase(24, False, "H100", H100, job_days=30, lifetime_months=36)
    short = pricing.recommend_purchase(24, False, "H100", H100, job_days=30, lifetime_months=12)
    assert (long["tier"], long["term"]) == ("reserved", "3yr")
    assert (short["tier"], short["term"]) == ("reserved", "1yr")


def test_short_jobs_cannot_reserve_and_interrupt_rate_matters():
    rec = pricing.recommend_purchase(20, False, "H100", H100, job_days=14)
    assert rec["tier"] == "on_demand"                      # 14-day job: no commitment
    a10g = {"on_demand_hr": 1.0, "spot_hr": 0.4, "reserved_1yr_hr": 0.8, "reserved_3yr_hr": 0.6}
    calm = pricing.recommend_purchase(8, True, "H100", H100)["options"]["spot"] / 1.5
    noisy = pricing.recommend_purchase(8, True, "A10G", a10g)["options"]["spot"] / 0.4
    assert noisy > calm                                    # more reclaims -> more rework overhead


# --- Ext 2: MBU right-sizing ----------------------------------------------------
def test_rightsize_respects_vram_bw_and_flops():
    cat = catalog_by_type()
    small = {"gpu_type": "A10G", "peak_mem_gb": 20.4, "peak_bw_tbs": 0.21, "peak_tflops": 37}
    assert m1_efficiency_audit.rightsize(small, cat)[0] == "L4"
    big = {"gpu_type": "H100", "peak_mem_gb": 67, "peak_bw_tbs": 0.9, "peak_tflops": 225}
    assert m1_efficiency_audit.rightsize(big, cat)[0] == "H100"          # nothing cheaper fits
    assert m1_efficiency_audit.rightsize(big, cat, allow_cross_vendor=True)[0] == "MI300X"


# --- Ext 3: cache_is_worth_it ---------------------------------------------------
def test_cache_break_even():
    w = pricing.cache_write_cost_per_m(3.0, write_multiplier=1.25)      # 0.75 $/M
    assert abs(pricing.cache_break_even_reads(w, 3.0) - 0.75 / 2.7) < 1e-9
    assert pricing.cache_is_worth_it(1.0, w, 0.10, 3.0)
    assert not pricing.cache_is_worth_it(0.1, w, 0.10, 3.0)
    # storage fees make cheap models need many more reads
    ws = pricing.cache_write_cost_per_m(0.20, 1.0, storage_per_m_hr=1.0, ttl_hours=1)
    assert pricing.cache_break_even_reads(ws, 0.20) > 5


def test_gated_cache_never_cheaper_than_free_cache():
    r = m2_inference_levers.run(verbose=False)
    assert r["optimized_daily"] >= r["ungated_optimized_daily"]
    assert 60 <= r["savings_pct"] <= 95


# --- Ext 4: reasoning budget ----------------------------------------------------
def test_reasoning_budget_split_and_cap():
    rb = m2_inference_levers.run(verbose=False)["reasoning"]
    assert rb["reasoning_energy_pct"] > rb["reasoning_cost_pct"] > rb["reasoning_traffic_pct"]
    assert rb["usd_saved_daily"] > 0 and rb["wh_saved_daily"] > 0


# --- Ext 5: carbon-aware scheduling ---------------------------------------------
def test_carbon_scheduling_regions():
    r = m6_carbon_scheduling.run(verbose=False)
    assert r["cleanest"] == "europe-north1" and r["cheapest"] == "us-east-wa"
    assert r["cleanest_carbon_saved_kg"] > 0 and r["cleanest_carbon_saved_pct"] > 90
