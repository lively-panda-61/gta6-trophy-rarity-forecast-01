"""Unit tests for model.py (written first, TDD).

Covers: logit helpers, quantile construction, output shape/caps,
flat-history behaviour (practice cases), synthetic trending behaviour,
determinism, and edge cases (rate 0 / 100, null rates, short history).
Run: python3 -m unittest discover -s tests -v
"""
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import (
    clamp_pct,
    forecast_case,
    forecast_trophy,
    inv_logit,
    logit,
    pinball_loss,
    series_for_trophy,
)

KIT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "gta6-trophy-rarity-forecast-01-kit")
INPUT = os.path.join(KIT, "input")

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def load_case(name):
    with open(os.path.join(INPUT, name + ".in")) as fh:
        return json.load(fh)


def make_case(history, horizon_minutes=90, at="2026-11-19T03:00:00Z",
              target="2026-11-19T04:30:00Z"):
    """Build a minimal case dict. history: list of dicts with slot/at/
    players_tracked/rates (rates in trophy order)."""
    trophies = [{"id": i + 1, "name": "T%d" % (i + 1), "description": "",
                 "tier": "bronze", "points": 15, "secret": False}
                for i in range(len(history[0]["rates"]))]
    return {"schema": "makefun/gta6-trophies-case/v1", "id": "syn",
            "at": at, "horizon_minutes": horizon_minutes, "target": target,
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "GTA VI", "platform": "PS5"},
            "trophies": trophies, "history": history, "notes": ""}


def read(slot, players, rates):
    return {"slot": slot, "at": slot, "players_tracked": players,
            "trophies_earned": 0, "rates": rates}


class TestHelpers(unittest.TestCase):
    def test_logit_roundtrip(self):
        for pct in (0.01, 0.63, 1.1, 25.0, 50.0, 99.0, 99.99):
            self.assertAlmostEqual(inv_logit(logit(pct)), pct, places=6)

    def test_clamp(self):
        self.assertEqual(clamp_pct(0.0), 0.01)
        self.assertEqual(clamp_pct(100.0), 99.99)
        self.assertEqual(clamp_pct(-5.0), 0.01)
        self.assertEqual(clamp_pct(150.0), 99.99)
        self.assertEqual(clamp_pct(1.1), 1.1)

    def test_pinball(self):
        # Equal quantiles score the mean absolute logit error.
        y, q = 22.0, 20.0
        loss = pinball_loss(y, q, 0.5)
        self.assertAlmostEqual(loss, abs(logit(y) - logit(q)), places=9)
        # Below the rate at tau=0.05 costs 2*tau per unit short.
        loss05 = pinball_loss(22.0, 20.0, 0.05)
        self.assertAlmostEqual(loss05, 2 * 0.05 * (logit(22.0) - logit(20.0)),
                               places=9)
        # Above the rate at tau=0.95 costs 2*(1-tau) per unit over.
        loss95 = pinball_loss(8.0, 20.0, 0.95)
        self.assertAlmostEqual(loss95, 2 * 0.05 * (logit(20.0) - logit(8.0)),
                               places=9)


class TestFlatHistory(unittest.TestCase):
    """Practice cases: completely flat histories (638 players, same rates)."""

    def test_practice_medians_near_last_rates(self):
        for name in ("p00", "p01"):
            case = load_case(name)
            out = forecast_case(case)
            last = case["history"][-1]["rates"]
            for row, last_rate in zip(out["trophies"], last):
                q = row["q"]
                self.assertEqual(len(q), 5)
                self.assertTrue(all(q[i] <= q[i + 1] for i in range(4)),
                                "quantiles must be non-decreasing")
                self.assertTrue(all(0 <= v <= 100 for v in q))
                if last_rate >= 99.99:
                    # The 100% trophy: median stays ~100 (a refresh could
                    # in principle dilute it, so a hair below is honest).
                    self.assertGreaterEqual(q[2], 99.9)
                else:
                    # Median must not sit far above the last rate: a refresh
                    # can only dilute (new players hold fewer trophies).
                    self.assertLessEqual(q[2], last_rate * 1.05 + 0.05)
                    # ... and not collapse to zero either.
                    self.assertGreaterEqual(q[2], last_rate * 0.3)

    def test_practice_output_shape(self):
        case = load_case("p00")
        out = forecast_case(case)
        self.assertEqual(set(out.keys()), {"trophies"})
        self.assertEqual(len(out["trophies"]), 55)
        ids = [r["id"] for r in out["trophies"]]
        self.assertEqual(sorted(ids), list(range(1, 56)))
        self.assertEqual(ids, list(range(1, 56)))
        raw = json.dumps(out, separators=(",", ":")).encode()
        self.assertLessEqual(len(raw), 65536)

    def test_determinism(self):
        case = load_case("p00")
        self.assertEqual(json.dumps(forecast_case(case), sort_keys=True),
                         json.dumps(forecast_case(case), sort_keys=True))


class TestTrending(unittest.TestCase):
    def test_rising_rates_median_rises(self):
        # Rates climbing steadily: median must sit above the last rate.
        hist = []
        for h in range(8):
            slot = "2026-11-19T%02d:00:00Z" % h
            hist.append(read(slot, 1000 + 100 * h,
                             [1.0 + 0.2 * h, 50.0 + 1.0 * h]))
        case = make_case(hist, horizon_minutes=24 * 60,
                         at="2026-11-19T08:00:00Z",
                         target="2026-11-20T08:00:00Z")
        out = forecast_case(case)
        self.assertGreater(out["trophies"][0]["q"][2], 2.4)
        self.assertGreater(out["trophies"][1]["q"][2], 57.0)

    def test_falling_rates_median_falls(self):
        # Dilution: players flooding in, rates falling.
        hist = []
        for h in range(8):
            slot = "2026-11-19T%02d:00:00Z" % h
            hist.append(read(slot, 1000 * (2 ** h),
                             [5.0 / (1 + 0.3 * h), 40.0 - 2.0 * h]))
        case = make_case(hist, horizon_minutes=24 * 60,
                         at="2026-11-19T08:00:00Z",
                         target="2026-11-20T08:00:00Z")
        out = forecast_case(case)
        self.assertLess(out["trophies"][0]["q"][2], 5.0)
        self.assertLess(out["trophies"][1]["q"][2], 40.0)

    def test_short_history_falls_back(self):
        hist = [read("2026-11-19T00:00:00Z", 638, [1.1, 50.0])]
        case = make_case(hist)
        out = forecast_case(case)
        for row in out["trophies"]:
            self.assertTrue(all(0 <= v <= 100 for v in row["q"]))
            self.assertTrue(all(row["q"][i] <= row["q"][i + 1]
                                for i in range(4)))

    def test_null_rates_skipped(self):
        hist = [
            read("2026-11-19T00:00:00Z", 638, [1.0, None]),
            read("2026-11-19T00:30:00Z", 700, [1.2, 50.0]),
            read("2026-11-19T01:00:00Z", 800, [1.4, 51.0]),
        ]
        case = make_case(hist)
        out = forecast_case(case)
        self.assertEqual(len(out["trophies"]), 2)
        self.assertTrue(all(0 <= v <= 100 for v in out["trophies"][1]["q"]))

    def test_series_extraction(self):
        case = load_case("p00")
        s = series_for_trophy(case, 0)
        self.assertEqual(len(s), 7)
        self.assertEqual(s[0][2], 0.63)  # (hours, players, rate)
        self.assertEqual(s[0][1], 638)


if __name__ == "__main__":
    unittest.main()
