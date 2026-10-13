"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45,
                   gpu_type: str | None = None, job_days: float | None = None,
                   prices: dict | None = None, lifetime_months: float = 36.0) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    Without `gpu_type` + `prices` this is the DOCUMENTED simple policy (v1):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    With them, it delegates to the cost-based v2 policy `recommend_purchase()`
    (per-GPU spot interruption rate + 1yr vs 3yr term + idle commitment hours).
    """
    if gpu_type is not None and prices is not None:
        return recommend_purchase(hours_per_day, interruptible, gpu_type, prices,
                                  job_days=job_days, lifetime_months=lifetime_months)["tier"]
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


# --- Extension 1: cost-based tier policy -----------------------------------------
# Per-hour probability a spot instance is reclaimed (illustrative 2026). Scarce,
# high-end parts are reclaimed less often on neoclouds than commodity inference GPUs.
SPOT_INTERRUPT_RATE = {
    "H100": 0.03, "H200": 0.03, "B200": 0.04, "MI300X": 0.05,
    "A100": 0.05, "A10G": 0.10, "L4": 0.08,
}
MIN_COMMIT_DAYS = 28  # a commitment only makes sense for a workload that recurs every month


def reserved_effective_hr(rate_hr: float, hours_per_day: float, term_months: float,
                          lifetime_months: float) -> float:
    """$ per *used* GPU-hour of a reservation.

    A reservation bills 24h/day for the whole term, so you pay for idle hours
    (24 / hours_per_day) and for months after the workload is retired
    (term / lifetime, when the workload dies before the term ends).
    """
    if hours_per_day <= 0:
        return float("inf")
    idle_mult = 24.0 / min(24.0, hours_per_day)
    stranded_mult = max(1.0, term_months / max(lifetime_months, 1e-9))
    return rate_hr * idle_mult * stranded_mult


def term_break_even_months(reserved_1yr_hr: float, reserved_3yr_hr: float) -> float:
    """Workload lifetime (months) above which 3yr beats 1yr: 36 * r3 / r1."""
    return 36.0 * reserved_3yr_hr / reserved_1yr_hr if reserved_1yr_hr > 0 else float("inf")


def recommend_purchase(hours_per_day: float, interruptible: bool, gpu_type: str, prices: dict,
                       job_days: float | None = None, lifetime_months: float = 36.0) -> dict:
    """v2 policy: price every eligible option per used GPU-hour and take the cheapest.

    `prices` holds on_demand_hr / spot_hr / reserved_1yr_hr / reserved_3yr_hr.
    - spot: only if interruptible and not 24/7; priced with the GPU-specific
      interruption rate via `spot_checkpoint_cost` (checkpoint + rework overhead).
    - reserved 1yr / 3yr: only if the job recurs all month (job_days >= 28);
      priced with `reserved_effective_hr` (idle hours + stranded months).
    Returns {'tier', 'term', 'effective_hr', 'interrupt_rate', 'options'}.
    """
    od = float(prices["on_demand_hr"])
    rate = SPOT_INTERRUPT_RATE.get(gpu_type, 0.05)
    options = {"on_demand": od}
    if interruptible and hours_per_day < 24:
        sim = spot_checkpoint_cost(100.0, float(prices["spot_hr"]), od, interrupt_rate=rate)
        options["spot"] = sim["spot_cost"] / 100.0
    if job_days is None or job_days >= MIN_COMMIT_DAYS:
        options["reserved_1yr"] = reserved_effective_hr(float(prices["reserved_1yr_hr"]), hours_per_day, 12, lifetime_months)
        options["reserved_3yr"] = reserved_effective_hr(float(prices["reserved_3yr_hr"]), hours_per_day, 36, lifetime_months)
    best = min(options, key=lambda k: (options[k], k != "on_demand"))  # ties -> stay flexible
    tier, _, term = best.partition("_") if best.startswith("reserved") else (best, "", "")
    return {"tier": tier, "term": term, "effective_hr": round(options[best], 4),
            "interrupt_rate": rate, "options": {k: round(v, 4) for k, v in options.items()}}


# --- Extension 3: is prompt caching actually worth it? ---------------------------
def cache_write_cost_per_m(price_in_per_m: float, write_multiplier: float = 1.25,
                           storage_per_m_hr: float = 0.0, ttl_hours: float = 5 / 60) -> float:
    """Extra $ per 1M prefix tokens to create one cache entry vs. not caching.

    Anthropic-style: write billed at 1.25x input, no storage fee (5-min TTL).
    Gemini-style: no write premium but a storage fee per 1M tokens per hour of TTL.
    """
    return price_in_per_m * (write_multiplier - 1.0) + storage_per_m_hr * ttl_hours


def cache_break_even_reads(write_cost_per_m: float, price_in_per_m: float,
                           read_discount: float = 0.10) -> float:
    """Reads per cache write needed for the read savings to repay the write cost."""
    saving_per_read = price_in_per_m * (1.0 - read_discount)
    return write_cost_per_m / saving_per_read if saving_per_read > 0 else float("inf")


def cache_is_worth_it(avg_cache_reads: float, write_cost_per_m: float,
                      read_discount: float = 0.10, price_in_per_m: float = 1.0) -> bool:
    """Cache only saves money when total savings from reads > cost of the write.

    Each read of a cached prefix saves (1 - read_discount) x input price; the write
    costs `write_cost_per_m` extra. Worth it iff reads exceed the break-even count.
    """
    return avg_cache_reads > cache_break_even_reads(write_cost_per_m, price_in_per_m, read_discount)


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
