"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Extension 3: prompt caching is only billed as a saving for (project, tier) prefixes
whose observed reads-per-write clear `pricing.cache_is_worth_it()`; the cache-write
premium is charged where caching stays on.
Extension 4: reasoning budget — $ and Wh split for is_reasoning traffic, plus a
routing-rule simulation that caps reasoning per team.

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Cache economics scenarios (illustrative 2026). The first one is the official M2 number.
CACHE_SCENARIOS = {
    "anthropic-5min": {"write_multiplier": 1.25, "storage_per_m_hr": 0.0, "ttl_min": 5},
    "gemini-60min": {"write_multiplier": 1.00, "storage_per_m_hr": 1.00, "ttl_min": 60},
}
REASONING_CAP_PER_TEAM = 0.10   # proposed rule: at most 10% of a team's requests may use reasoning
REASONING_OUTPUT_TAX = 6        # reasoning requests emit ~6x output tokens (thinking tokens)


def _minute(ts: str) -> int:
    hh, mm = ts[11:13], ts[14:16]
    return int(hh) * 60 + int(mm)


def cache_policy(rows: list[dict], scenario: str = "anthropic-5min") -> dict:
    """Decide caching per (project, tier) prefix from observed reads per cache write.

    Requests of one prefix inside one TTL window share a single cache entry: the first
    writes it, the rest read it. avg_reads = mean(n_requests_in_window - 1).
    """
    sc = CACHE_SCENARIOS[scenario]
    windows = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = (r["project"], r["route_tier"])
        windows[key][_minute(r["ts"]) // sc["ttl_min"]] += 1
    policy = {}
    for key, w in windows.items():
        tier = key[1]
        price_in = MODEL_PRICES[tier][0]
        write_cost = pricing.cache_write_cost_per_m(price_in, sc["write_multiplier"],
                                                    sc["storage_per_m_hr"], sc["ttl_min"] / 60)
        avg_reads = sum(n - 1 for n in w.values()) / len(w)
        policy[key] = {
            "avg_reads": round(avg_reads, 2), "writes": len(w),
            "break_even_reads": round(pricing.cache_break_even_reads(write_cost, price_in), 2),
            "write_cost_per_m": write_cost,
            "worth_it": pricing.cache_is_worth_it(avg_reads, write_cost, 0.10, price_in),
        }
    return policy


def optimized_cost(rows: list[dict], policy: dict | None, cascade: bool = True,
                   cache: bool = True, batch: bool = True) -> float:
    """Cascade + batch + caching. With a policy, caching is gated and writes are paid for.

    The flags switch levers off one at a time for the per-lever breakdown.
    """
    cost = 0.0
    prefix = defaultdict(list)
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"])) if cache else 0
        is_batch = bool(int(num(r["is_batch"]))) and batch
        tier = r["route_tier"] if cascade else "large"
        pin, pout = MODEL_PRICES[tier]
        key = (r["project"], r["route_tier"])
        if policy is not None and not policy[key]["worth_it"]:
            cached = 0
        cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        prefix[key].append(cached)
    if policy is not None:
        for key, p in policy.items():
            if p["worth_it"] and prefix[key] and cache:
                mean_prefix = sum(prefix[key]) / len(prefix[key])
                # each write: pay the write cost instead of the 0.1x read rate already billed
                extra_per_m = p["write_cost_per_m"] + MODEL_PRICES[key[1]][0] * (1 - 0.10)
                cost += p["writes"] * (mean_prefix / 1e6) * extra_per_m
    return cost


def reasoning_budget(rows: list[dict], cap: float = REASONING_CAP_PER_TEAM) -> dict:
    """Split $ and Wh by is_reasoning, then simulate capping reasoning per team.

    Routing rule: keep reasoning only for the most complex requests of a team (proxy:
    longest input context), up to `cap` of that team's traffic; demoted requests are
    answered without thinking tokens (output / REASONING_OUTPUT_TAX).
    """
    def price(r, out=None, reasoning=None):
        inp = int(num(r["input_tokens"]))
        out = int(num(r["output_tokens"])) if out is None else out
        reasoning = bool(int(num(r["is_reasoning"]))) if reasoning is None else reasoning
        pin, pout = MODEL_PRICES[r["route_tier"]]
        usd = pricing.request_cost(inp, out, pin, pout, cached_in=int(num(r["cached_input_tokens"])),
                                   batch=bool(int(num(r["is_batch"]))))
        return usd, sustainability.wh_per_query(inp + out, is_reasoning=reasoning)

    split = {0: {"n": 0, "usd": 0.0, "wh": 0.0, "tokens": 0}, 1: {"n": 0, "usd": 0.0, "wh": 0.0, "tokens": 0}}
    by_team = defaultdict(list)
    for r in rows:
        k = int(num(r["is_reasoning"]))
        usd, wh = price(r)
        s = split[k]
        s["n"] += 1; s["usd"] += usd; s["wh"] += wh
        s["tokens"] += int(num(r["input_tokens"])) + int(num(r["output_tokens"]))
        by_team[r["team"]].append(r)

    tot_n = split[0]["n"] + split[1]["n"]
    tot_usd = split[0]["usd"] + split[1]["usd"]
    tot_wh = split[0]["wh"] + split[1]["wh"]

    # cap simulation
    capped_usd, capped_wh, demoted = 0.0, 0.0, 0
    team_share = {}
    for team, rs in by_team.items():
        reasoning = sorted([r for r in rs if int(num(r["is_reasoning"]))],
                           key=lambda r: -int(num(r["input_tokens"])))
        team_share[team] = round(len(reasoning) / len(rs), 3)
        allowed = int(cap * len(rs))
        keep = {id(r) for r in reasoning[:allowed]}
        min_kept_input = int(num(reasoning[allowed - 1]["input_tokens"])) if 0 < allowed <= len(reasoning) else None
        for r in rs:
            if int(num(r["is_reasoning"])) and id(r) not in keep:
                demoted += 1
                usd, wh = price(r, out=max(1, int(num(r["output_tokens"])) // REASONING_OUTPUT_TAX), reasoning=False)
            else:
                usd, wh = price(r)
            capped_usd += usd; capped_wh += wh
        if team == "eval":
            eval_threshold = min_kept_input

    per_req = {k: {"usd": v["usd"] / v["n"], "wh": v["wh"] / v["n"]} for k, v in split.items() if v["n"]}
    return {
        "reasoning_traffic_pct": round(split[1]["n"] / tot_n * 100, 1),
        "reasoning_cost_pct": round(split[1]["usd"] / tot_usd * 100, 1),
        "reasoning_energy_pct": round(split[1]["wh"] / tot_wh * 100, 1),
        "usd": {k: round(v["usd"], 2) for k, v in split.items()},
        "wh": {k: round(v["wh"], 1) for k, v in split.items()},
        "wh_per_req_ratio": round(per_req[1]["wh"] / per_req[0]["wh"], 1),
        "usd_per_req_ratio": round(per_req[1]["usd"] / per_req[0]["usd"], 2),
        "team_reasoning_share": team_share,
        "cap": cap, "demoted": demoted, "eval_min_input_tokens": eval_threshold,
        "capped_usd": round(capped_usd, 2), "capped_wh": round(capped_wh, 1),
        "usd_saved_daily": round(tot_usd - capped_usd, 2), "wh_saved_daily": round(tot_wh - capped_wh, 1),
        "total_usd": round(tot_usd, 2), "total_wh": round(tot_wh, 1),
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

    # OPTIMIZED: cascade (route_tier), prompt caching (gated, Ext 3), batch API
    ungated_cost = optimized_cost(rows, policy=None)
    policies = {name: cache_policy(rows, name) for name in CACHE_SCENARIOS}
    gated = {name: optimized_cost(rows, p) for name, p in policies.items()}
    opt_cost = gated["anthropic-5min"]

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    rb = reasoning_budget(rows)

    # per-lever breakdown, applied in order: cascade -> cache (gated) -> batch
    pol = policies["anthropic-5min"]
    after_cascade = optimized_cost(rows, pol, cache=False, batch=False)
    after_cache = optimized_cost(rows, pol, batch=False)
    lever_daily = {"cascade": base_cost - after_cascade, "cache": after_cascade - after_cache,
                   "batch": after_cache - opt_cost}
    small_share = sum(r["route_tier"] == "small" for r in rows) / len(rows)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("per-lever $/day (applied in order): " + ", ".join(
            f"{k} ${v:,.2f} ({v / base_cost * 100:.1f}%)" for k, v in lever_daily.items())
            + f"   [{small_share:.0%} of requests routed to small]")

        print("\n-- Ext 3: is caching worth it? (reads per cache write vs break-even) --")
        print(f"{'scenario':16}{'project':14}{'tier':7}{'writes':>7}{'avg reads':>10}{'break-even':>11}  decision")
        for name, pol in policies.items():
            for (proj, tier), p in sorted(pol.items()):
                print(f"{name:16}{proj or '(untagged)':14}{tier:7}{p['writes']:>7}{p['avg_reads']:>10}{p['break_even_reads']:>11}  "
                      f"{'CACHE ON' if p['worth_it'] else 'cache off'}")
        print(f"optimized $/day: ungated (every cached token discounted, writes free) ${ungated_cost:,.2f}")
        for name, c in gated.items():
            print(f"                 gated [{name}] ${c:,.2f}  ({pricing.dollars_per_million(c, total_tokens):.3f} $/1M)")

        print("\n-- Ext 4: reasoning budget --")
        print(f"reasoning = {rb['reasoning_traffic_pct']}% of requests, {rb['reasoning_cost_pct']}% of $, "
              f"{rb['reasoning_energy_pct']}% of Wh")
        print(f"  non-reasoning: ${rb['usd'][0]:,.2f}/day  {rb['wh'][0]:,.1f} Wh/day")
        print(f"  reasoning    : ${rb['usd'][1]:,.2f}/day  {rb['wh'][1]:,.1f} Wh/day")
        print(f"  per request  : reasoning costs {rb['usd_per_req_ratio']}x $ and {rb['wh_per_req_ratio']}x Wh")
        print(f"  reasoning share by team: {rb['team_reasoning_share']}")
        print(f"rule: reasoning only for top {rb['cap']:.0%} of each team's requests by context length "
              f"(eval: input >= {rb['eval_min_input_tokens']} tokens)")
        print(f"  -> demotes {rb['demoted']} requests, saves ${rb['usd_saved_daily']:,.2f}/day "
              f"({rb['usd_saved_daily'] / rb['total_usd'] * 100:.1f}%) and {rb['wh_saved_daily']:,.0f} Wh/day "
              f"({rb['wh_saved_daily'] / rb['total_wh'] * 100:.1f}%)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "ungated_optimized_daily": round(ungated_cost, 2),
        "cache_policy": policies, "gated_daily": {k: round(v, 2) for k, v in gated.items()},
        "reasoning": rb, "lever_daily": {k: round(v, 2) for k, v in lever_daily.items()},
        "small_share": round(small_share, 3),
    }


if __name__ == "__main__":
    run()
