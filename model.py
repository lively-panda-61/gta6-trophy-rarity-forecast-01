"""Token-rarity forecaster, model "counts-2" (written from scratch).

Idea: an Exophase rate is a ratio, rate = earners / players-tracked.
Both sides move between the checkpoint and the target, and they move for
different reasons -- new players join (diluting rates when they have not
earned a trophy yet) while the existing cohort keeps earning. So the
forecast works in counts, not in rates alone:

  * players N(t): fit log N against log(1+t) and against t on the recent
    refresh points, damp the growth, extrapolate to the target;
  * earners E_i(t) per trophy: fit log(E_i + 0.5) against log(1+t) with
    recency weights, extrapolate with the same damping, and blend with
    "E grows with N" (the rate stays put) by fit strength;
  * median rate = E_hat / N_hat, combined in logit space with the
    log-time logit-trend estimate and with the last rate, weights from
    each fit's residual;
  * spread (logit units) = random-walk floor + fit residual + slope
    uncertainty + a refresh-jump allowance estimated from the jumps the
    history actually shows (with a prior when the history is flat: the
    page refreshes every few hours and a refresh dilutes rates).

Everything is deterministic: no randomness, no wall-clock use, trophies
processed in id order. Standard library only.
"""
import math

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE_HOURS_EPOCH = "2026-11-19T00:00:00Z"

# Tunables (chosen by backtest, see tests/test_backtest.py).
RECENCY_HALF_LIFE_H = 48.0   # recency weight half-life for fits
GROWTH_DAMP = 0.85           # damping on extrapolated log growth
SPREAD_FLOOR = 0.5           # logit spread floor
SPREAD_CAP = 2.5             # logit spread cap
RW_SIGMA = 0.5               # random-walk sigma per sqrt(day)
REFRESH_PRIOR_JUMP = 1.0     # typical logit drop of one refresh when no
                             # refresh was ever observed (launch night: the
                             # first refresh diluted rates hard as N jumped
                             # 638 -> 3,868 and new players held ~6% of the
                             # old earn rates)
REFRESH_PRIOR_HOURS = 6.0    # prior mean interval between refreshes


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


def parse_hours(ts):
    """ISO UTC timestamp -> hours since 2026-11-19T00:00:00Z."""
    from datetime import datetime
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    base = datetime.fromisoformat("2026-11-19T00:00:00+00:00")
    return (dt - base).total_seconds() / 3600.0


def _wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std, se_b)."""
    sw = sum(ws)
    if sw <= 0:
        return ys[0] if ys else 0.0, 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        resid = [y - my for y in ys]
        s2 = sum(w * r * r for w, r in zip(ws, resid)) / sw
        return my, 0.0, math.sqrt(max(s2, 0.0)), 0.0
    slope = sxy / sxx
    inter = my - slope * mx
    resid = [y - (inter + slope * x) for x, y in zip(xs, ys)]
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / sw
    se = math.sqrt(max(s2, 0.0) / sxx)
    return inter, slope, math.sqrt(max(s2, 0.0)), se


def series_for_trophy(case, idx):
    """(hours, players, rate) for one trophy, non-null reads only."""
    out = []
    for rd in case["history"]:
        v = rd["rates"][idx]
        if v is None:
            continue
        out.append((parse_hours(rd["at"]), float(rd["players_tracked"]),
                    float(v)))
    return out


def _refresh_points(series):
    """Dedupe a series to distinct page states (the page is static between
    refreshes, so identical consecutive (players, rate) pairs carry no new
    information). Returns list of (t, N, rate, E)."""
    pts = []
    prev = None
    for t, n, r in series:
        key = (n, r)
        if key == prev:
            continue
        prev = key
        pts.append((t, n, r, r / 100.0 * n))
    return pts


def _forecast_series_value(pts, t_target, value_fn):
    """Extrapolate a positive series value to t_target. Averages two
    estimators in log space (log value vs ln(1+t), and log value vs t over
    the last 24h), both damped: a log-log fit alone badly under-shoots a
    process that grows linearly in t over a long horizon. Returns
    (value_hat, resid_std, slope_se, n_points) or None."""
    if not pts:
        return None
    t_last = pts[-1][0]
    xs, ys, ws = [], [], []
    for t, n, r, e in pts:
        v = value_fn(t, n, r, e)
        if v is None or v <= 0:
            continue
        xs.append(math.log1p(max(t, 0.0)))
        ys.append(math.log(v))
        ws.append(math.exp(-(t_last - t) * math.log(2) / RECENCY_HALF_LIFE_H))
    if len(xs) < 2:
        return None
    a, b, resid, se = _wls(xs, ys, ws)
    x_t = math.log1p(max(t_target, 0.0))
    x_last = math.log1p(max(t_last, 0.0))
    ests = [a + b * x_last + GROWTH_DAMP * b * (x_t - x_last)]
    recent = [(t, value_fn(t, n, r, e)) for t, n, r, e in pts
              if t_last - t <= 24.0 and value_fn(t, n, r, e) > 0]
    if len(recent) >= 2:
        xs2 = [t for t, _ in recent]
        ys2 = [math.log(v) for _, v in recent]
        a2, b2, _, _ = _wls(xs2, ys2, [1.0] * len(recent))
        ests.append(a2 + b2 * t_last + GROWTH_DAMP * b2 * (t_target - t_last))
    log_hat = sum(ests) / len(ests)
    return math.exp(log_hat), resid, se, len(xs)


def _fit_log_growth(pts, t_target, value_fn):
    """Back-compat wrapper: (log_value_hat, resid, se, n)."""
    out = _forecast_series_value(pts, t_target, value_fn)
    if out is None:
        return None
    v, resid, se, k = out
    return math.log(v), resid, se, k


def _forecast_n(case, t_target):
    """Forecast players-tracked at the target. Two estimators (log-log and
    log-linear in t, both recency weighted) averaged in log space."""
    series = []
    for rd in case["history"]:
        series.append((parse_hours(rd["at"]), float(rd["players_tracked"])))
    pts = []
    prev = None
    for t, n in series:
        if n == prev:
            continue
        prev = n
        pts.append((t, n))
    if len(pts) < 2:
        return pts[-1][1] if pts else 0.0, 0.0
    t_last = pts[-1][0]
    # Estimator 1: log N ~ a + b*ln(1+t).
    ests = []
    xs = [math.log1p(max(t, 0.0)) for t, _ in pts]
    ys = [math.log(n) for _, n in pts]
    ws = [math.exp(-(t_last - t) * math.log(2) / RECENCY_HALF_LIFE_H)
          for t, _ in pts]
    a, b, resid, _ = _wls(xs, ys, ws)
    x_t = math.log1p(max(t_target, 0.0))
    x_last = math.log1p(max(t_last, 0.0))
    ests.append(a + b * x_last + GROWTH_DAMP * b * (x_t - x_last))
    # Estimator 2: log N ~ a + b*t over the last 24h, damped.
    recent = [(t, n) for t, n in pts if t_last - t <= 24.0]
    if len(recent) >= 2:
        xs2 = [t for t, _ in recent]
        ys2 = [math.log(n) for _, n in recent]
        ws2 = [1.0] * len(recent)
        a2, b2, _, _ = _wls(xs2, ys2, ws2)
        ests.append(a2 + b2 * t_last + GROWTH_DAMP * b2 * (t_target - t_last))
    log_n = sum(ests) / len(ests)
    n_hat = math.exp(log_n)
    # Never forecast fewer players than the last read saw.
    return max(n_hat, pts[-1][1]), resid


def _median_estimate(case, idx, pts, n_hat, t_eval, t_last, r_last):
    """Combined median (logit) of the counts and trend estimates evaluated
    at t_eval, plus the last rate. Returns (median_logit, resid, se)."""
    candidates = []
    fit_e = _fit_log_growth(pts, t_eval, lambda t, n, r, e: e + 0.5)
    if fit_e is not None:
        log_e, resid_e, se_e, k = fit_e
        e_hat = math.exp(log_e)
        e_grow = (pts[-1][3] + 0.5) * n_hat / max(pts[-1][1], 1.0)
        strength = min(1.0, k / 12.0) * (1.0 if resid_e < 0.4 else 0.5)
        e_blend = math.exp(strength * math.log(max(e_hat, 0.5)) +
                           (1 - strength) * math.log(max(e_grow, 0.5)))
        r_counts = 100.0 * e_blend / max(n_hat, 1.0)
        candidates.append((logit(r_counts), strength + 0.25, resid_e, se_e))
    recent = [(t, r) for t, n, r, e in pts if t_last - t <= 24.0]
    if len(recent) >= 3:
        xs = [math.log1p(max(t, 0.0)) for t, _ in recent]
        ys = [logit(r) for _, r in recent]
        ws = [math.exp(-(t_last - t) * math.log(2) / RECENCY_HALF_LIFE_H)
              for t, _ in recent]
        a, b, resid_t, se_t = _wls(xs, ys, ws)
        med_trend = a + b * math.log1p(max(t_eval, 0.0))
        candidates.append((med_trend, 1.0, resid_t, se_t))
    candidates.append((logit(r_last), 0.75, 0.0, 0.0))
    num = 0.0
    den = 0.0
    worst_resid = 0.0
    worst_se = 0.0
    for med, w, resid, se in candidates:
        wgt = w / (1.0 + 4.0 * resid * resid)
        num += wgt * med
        den += wgt
        worst_resid = max(worst_resid, resid)
        worst_se = max(worst_se, se)
    med_logit = num / den if den > 0 else logit(r_last)
    return med_logit, worst_resid, worst_se


def _refresh_stats(pts, t_at):
    """(typical interval hours, staleness hours, typical jump logit).
    Interval prior 6h when fewer than 2 gaps are observed."""
    gaps = [pts[j][0] - pts[j - 1][0] for j in range(1, len(pts))]
    gaps = [g for g in gaps if g > 0]
    if len(gaps) >= 2:
        gaps.sort()
        interval = gaps[len(gaps) // 2]
    elif len(gaps) == 1:
        interval = gaps[0]
    else:
        interval = REFRESH_PRIOR_HOURS
    interval = min(max(interval, 1.0), 24.0)
    staleness = max(0.0, t_at - pts[-1][0]) if pts else 0.0
    jumps = [abs(logit(pts[j][2]) - logit(pts[j - 1][2]))
             for j in range(1, len(pts))]
    if jumps:
        jumps.sort()
        jump = jumps[len(jumps) // 2]
    else:
        jump = REFRESH_PRIOR_JUMP
    return interval, staleness, jump


def _mixture_quantile(m_a, s_a, m_b, s_b, p, tau):
    """tau-quantile of the two-component normal mixture in logit space:
    (1-p) * N(m_a, s_a) + p * N(m_b, s_b). Bisection on the CDF."""
    from math import erf, sqrt

    def cdf(x):
        def phi(z):
            return 0.5 * (1.0 + erf(z / sqrt(2.0)))
        return (1.0 - p) * phi((x - m_a) / max(s_a, 1e-6)) + \
            p * phi((x - m_b) / max(s_b, 1e-6))

    lo = min(m_a - 8 * max(s_a, 0.05), m_b - 8 * max(s_b, 0.05))
    hi = max(m_a + 8 * max(s_a, 0.05), m_b + 8 * max(s_b, 0.05))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_trophy(case, idx, n_hat):
    """Five quantiles (percent) for one trophy.

    The answer is the page as of the last refresh before the target, so
    the forecast is a mixture over whether a refresh lands inside the
    horizon: no refresh -> the answer is exactly the last read's rates
    (the page is static between refreshes); a refresh -> the model's
    prediction at the refresh time, which the counts/trend fits estimate.
    """
    t_target = parse_hours(case["target"])
    t_at = parse_hours(case["at"])
    days = case["horizon_minutes"] / 1440.0
    horizon_h = case["horizon_minutes"] / 60.0
    series = series_for_trophy(case, idx)
    if not series:
        return 0.01, SPREAD_FLOOR
    t_last, n_last, r_last = series[-1]
    pts = _refresh_points(series)
    n_pts = len(pts)

    interval, staleness, jump = _refresh_stats(pts, t_at)
    # Time of the next refresh, roughly uniform in a window around
    # (interval - staleness); overdue (staleness >= interval) means
    # imminent. p = P(a refresh lands inside the horizon).
    lo_u = max(0.0, interval - staleness - 0.5 * interval)
    hi_u = max(0.5, interval - staleness + 0.5 * interval)
    if staleness >= interval:
        lo_u, hi_u = 0.0, 0.5 * interval
    p_refresh = min(0.95, max(0.0, (horizon_h - lo_u) / max(hi_u - lo_u, 1e-9)))
    if horizon_h >= 3.0 * interval:
        p_refresh = 0.95  # many refreshes fit in a long horizon

    # Component A: no refresh -> the answer is the last read, exactly.
    m_a = logit(r_last)
    s_a = 0.05
    # Component B: a refresh lands -> prediction at the refresh time,
    # roughly half an interval before the target (or mid-horizon when the
    # horizon is shorter than the interval).
    t_r = t_target - min(0.5 * interval, 0.5 * horizon_h)
    n_hat_r, _ = _forecast_n(case, t_r)
    m_b, resid_b, se_b = _median_estimate(case, idx, pts, n_hat_r, t_r,
                                          t_last, r_last)
    if n_pts < 3 and r_last < 99.0:
        # Flat history: no refresh was ever observed, so the fits cannot
        # know a refresh dilutes rates. A refresh drops rates by about
        # the prior jump (launch night: the mean rate fell ~25%).
        m_b -= jump
    s_rw = RW_SIGMA * math.sqrt(days)
    x_r = math.log1p(max(t_r, 0.0))
    x_last = math.log1p(max(t_last, 0.0))
    s_extra = se_b * max(0.0, x_r - x_last)
    s_b = math.sqrt(s_rw ** 2 + resid_b ** 2 + s_extra ** 2 +
                    (0.5 * jump) ** 2)
    s_b = min(SPREAD_CAP, max(0.15, s_b))

    qs = []
    for tau in LEVELS:
        ql = _mixture_quantile(m_a, s_a, m_b, s_b, p_refresh, tau)
        qs.append(clamp_pct(inv_logit(ql)))
    return qs


def forecast_case(case):
    """Full forecast dict for one case."""
    t_target = parse_hours(case["target"])
    n_hat, _ = _forecast_n(case, t_target)
    rows = []
    for idx, tr in enumerate(case["trophies"]):
        q = forecast_trophy(case, idx, n_hat)
        q = [round(v, 4) for v in q]
        # Enforce non-decreasing (rounding can tie, never invert).
        for j in range(1, 5):
            q[j] = max(q[j], q[j - 1])
        rows.append({"id": tr["id"], "q": q})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}
