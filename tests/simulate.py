"""Synthetic Exophase process simulator + pinball scoring + baselines.

Simulates what the house recorder sees: players join over time, earn
trophies with per-trophy hazards, and Exophase refreshes the page's counts
only every few hours (reads between refreshes show stale, identical values,
rates rounded to 2 decimals). The answering read is the page as of the last
refresh before the target, exactly as the rules describe.

Baselines implemented from the rules text:
  - house baseline: last rate, spread 0.5*sqrt(days) on the logit scale.
  - house curve baseline: least-squares logit-rate line vs ln(1+hours)
    over the last 24h of reads, spread = RW + residual.
  - v1-style: weighted log-time trend (recency weights), median blended
    toward the last rate when the fit is weak, spread = RW + residual +
    slope SE + 0.6 refresh allowance, floor 0.5 cap 2.5.
"""
import json
import math
import random

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = "2026-11-19T00:00:00Z"


def _parse(ts):
    # ISO UTC "2026-11-19T03:00:00Z" (or with fractional seconds) -> hours
    # since 2026-11-19T00:00:00Z.
    from datetime import datetime
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    base = datetime.fromisoformat("2026-11-19T00:00:00+00:00")
    return (dt - base).total_seconds() / 3600.0


def _fmt(hours):
    from datetime import datetime, timedelta
    base = datetime.fromisoformat("2026-11-19T00:00:00+00:00")
    return (base + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def pinball_loss(y, q, tau):
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_case(actual_rates, forecast):
    """actual_rates: list of true answer rates (percent) in trophy order.
    forecast: {"trophies": [{"id": n, "q": [...]}]}. Returns mean pinball
    loss over all (trophy, level) items."""
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = {r["id"]: r["q"] for r in forecast["trophies"]}
    total, n = 0.0, 0
    for i, y in enumerate(actual_rates):
        q = rows[i + 1]
        for tau, qv in zip(levels, q):
            total += pinball_loss(y, qv, tau)
            n += 1
    return total / n


# ---------------------------------------------------------------- simulator

def simulate(rng, n_trophies=55, checkpoint_h=48.0, horizon_h=108.5,
             growth_A=18.0, growth_B=1.1, refresh_h=5.0,
             earn_tau_lo=2.0, earn_tau_hi=120.0, n0=638.0,
             read_every_min=10.0):
    """Generate one synthetic case + its answer.

    Returns (case_dict, answer_rates). Player growth N(t) =
    n0 * (1 + t/A)^B for t >= 0 (t in hours since release); before release
    N = n0. Each trophy i has eventual earn probability p_i (log-spaced)
    and earn time constant tau_i; a player joining at s has earned by t
    with probability p_i * (1 - exp(-(t - s) / tau_i)). The page refreshes
    every ~refresh_h hours (jittered); reads between refreshes repeat the
    last refreshed values.
    """
    p = [10 ** (-3.0 + 3.0 * i / max(1, n_trophies - 1)) for i in range(n_trophies)]
    p[-1] = 1.0  # the "everyone earns it" trophy, like the 100% one
    tau = [earn_tau_lo * (earn_tau_hi / earn_tau_lo) ** (i / max(1, n_trophies - 1))
           for i in range(n_trophies)]

    def n_true(t):
        if t <= 0:
            return n0
        return n0 * (1.0 + t / growth_A) ** growth_B

    def e_true(i, t):
        # Earners of trophy i by time t: integrate over join times.
        # dN/ds = n0 * B/A * (1 + s/A)^(B-1); a joiner at s has earned by t
        # with prob p_i * (1 - exp(-(t-s)/tau_i)). Numeric integration.
        if t <= 0:
            return n0 * p[i]
        steps = 60
        total = 0.0
        ds = t / steps
        for k in range(steps):
            s = (k + 0.5) * ds
            dnds = n0 * (growth_B / growth_A) * (1.0 + s / growth_A) ** (growth_B - 1)
            total += dnds * p[i] * (1.0 - math.exp(-(t - s) / tau[i])) * ds
        # players present at t=0 (the launch-night cohort)
        total += n0 * p[i] * (1.0 - math.exp(-t / tau[i]))
        return total

    # Refresh schedule: first refresh at the first read (t=-1/6h), then
    # every refresh_h hours jittered by +-25%.
    refresh_times = [-1.0 / 6.0]
    t = -1.0 / 6.0
    while t < checkpoint_h + horizon_h + 2.0:
        t += refresh_h * rng.uniform(0.75, 1.25)
        refresh_times.append(t)

    def page_at(t):
        """True values as of the last refresh at or before t."""
        rt = max(r for r in refresh_times if r <= t)
        if rt in _page_cache:
            return _page_cache[rt]
        n = n_true(rt)
        rates = []
        for i in range(n_trophies):
            r = 100.0 * e_true(i, rt) / n
            rates.append(round(min(100.0, max(0.0, r)), 2))
        out = (int(round(n)), rates)
        _page_cache[rt] = out
        return out

    _page_cache = {}

    target_h = checkpoint_h + horizon_h
    # History: reads every read_every_min minutes from t=-0.17h to checkpoint.
    history = []
    t = -1.0 / 6.0
    while t < checkpoint_h:
        n, rates = page_at(t)
        history.append({"slot": _fmt(max(t, -1.0 / 6.0)), "at": _fmt(max(t, -1.0 / 6.0)),
                        "players_tracked": n, "trophies_earned": 0,
                        "rates": rates})
        t += read_every_min / 60.0
    trophies = [{"id": i + 1, "name": "T%d" % (i + 1), "description": "",
                 "tier": "bronze", "points": 15, "secret": False}
                for i in range(n_trophies)]
    case = {"schema": "makefun/gta6-trophies-case/v1", "id": "sim",
            "at": _fmt(checkpoint_h), "horizon_minutes": int(horizon_h * 60),
            "target": _fmt(target_h), "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "GTA VI", "platform": "PS5"},
            "trophies": trophies, "history": history, "notes": ""}
    # Answer: first read in [target, target+60min): page as of last refresh.
    _, answer = page_at(target_h)
    return case, answer


# ---------------------------------------------------------------- baselines

def _last_rates(case):
    hist = case["history"]
    n = len(case["trophies"])
    last = [None] * n
    for rd in reversed(hist):
        for i, v in enumerate(rd["rates"]):
            if last[i] is None and v is not None:
                last[i] = v
        if all(v is not None for v in last):
            break
    return [0.01 if v is None else v for v in last]


def house_baseline(case):
    days = case["horizon_minutes"] / 1440.0
    s = 0.5 * math.sqrt(days)
    rows = []
    for i, r in enumerate(_last_rates(case)):
        q = [round(clamp_pct(inv_logit(logit(r) + z * s)), 4) for z in Z]
        rows.append({"id": i + 1, "q": q})
    return {"trophies": rows}


def _wls(xs, ys, ws):
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    slope = sxy / sxx
    inter = my - slope * mx
    resid = [y - (inter + slope * x) for x, y in zip(xs, ys)]
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / max(sw, 1e-9)
    return inter, slope, math.sqrt(max(s2, 0.0))


def house_curve_baseline(case):
    hist = case["history"]
    n = len(case["trophies"])
    t_target = _parse(case["target"])
    days = case["horizon_minutes"] / 1440.0
    s_rw = 0.5 * math.sqrt(days)
    rows = []
    for i in range(n):
        pts = []
        for rd in hist:
            if rd["rates"][i] is None:
                continue
            t = _parse(rd["at"])
            if t_target - t <= 24.0:
                pts.append((math.log1p(max(t, 0.0)), logit(rd["rates"][i])))
        last = _last_rates(case)[i]
        if len(pts) < 3:
            med, s = logit(last), s_rw
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            inter, slope, resid = _wls(xs, ys, [1.0] * len(xs))
            med = inter + slope * math.log1p(max(t_target, 0.0))
            s = math.sqrt(s_rw ** 2 + resid ** 2)
        q = [round(clamp_pct(inv_logit(med + z * s)), 4) for z in Z]
        rows.append({"id": i + 1, "q": q})
    return {"trophies": rows}


def v1_style(case):
    """Reimplementation of the previous entry's approach (from its design
    notes): weighted log-time logit regression, recency weights
    exp(-age/48h), median blended toward the last rate when the fit is
    weak, spread = RW + residual + slope SE + 0.6 refresh allowance
    (floor 0.5, cap 2.5)."""
    hist = case["history"]
    n = len(case["trophies"])
    t_target = _parse(case["target"])
    t_last = _parse(hist[-1]["at"])
    days = case["horizon_minutes"] / 1440.0
    s_rw = 0.5 * math.sqrt(days)
    rows = []
    for i in range(n):
        pts = []
        for rd in hist:
            if rd["rates"][i] is None:
                continue
            t = _parse(rd["at"])
            pts.append((math.log1p(max(t, 0.0)), logit(rd["rates"][i]),
                        math.exp(-(t_last - t) / 48.0)))
        last = _last_rates(case)[i]
        if len(pts) < 3:
            med, s = logit(last), s_rw
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ws = [p[2] for p in pts]
            inter, slope, resid = _wls(xs, ys, ws)
            x_t = math.log1p(max(t_target, 0.0))
            med_fit = inter + slope * x_t
            # slope standard error (weighted)
            sw = sum(ws)
            mx = sum(w * x for w, x in zip(ws, xs)) / sw
            sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
            se = resid / math.sqrt(sxx) if sxx > 0 else 0.0
            # blend toward last rate when the fit is weak
            strength = min(1.0, len(pts) / 20.0) * (1.0 if resid < 0.5 else 0.5)
            med = strength * med_fit + (1 - strength) * logit(last)
            s = min(2.5, max(0.5, math.sqrt(s_rw ** 2 + resid ** 2 +
                                            (se * (x_t - mx)) ** 2 + 0.6 ** 2)))
        q = [round(clamp_pct(inv_logit(med + z * s)), 4) for z in Z]
        rows.append({"id": i + 1, "q": q})
    return {"trophies": rows}
