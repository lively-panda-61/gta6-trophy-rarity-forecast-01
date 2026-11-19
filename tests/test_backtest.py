"""Backtest: the new model must beat every house baseline and the v1-style
model on mean pinball loss across synthetic Exophase scenarios, at both
the final horizon (4.5 days) and the practice horizon (90 minutes).

Run: python3 -m unittest discover -s tests -v
"""
import os
import random
import sys
import unittest
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from model import forecast_case
from simulate import (house_baseline, house_curve_baseline, score_case,
                      simulate, v1_style)

SCENARIOS = [
    # (name, kwargs) -- growth_A/B: player growth; refresh_h: page refresh
    # interval; earn taus: how fast trophies are earned.
    ("launch-fast", dict(growth_A=8.0, growth_B=1.4, refresh_h=4.0)),
    ("launch-slow", dict(growth_A=30.0, growth_B=0.9, refresh_h=6.0)),
    ("steady", dict(growth_A=18.0, growth_B=1.1, refresh_h=5.0)),
    ("saturating", dict(growth_A=6.0, growth_B=0.6, refresh_h=8.0)),
    ("flat", dict(growth_A=1e9, growth_B=0.0, refresh_h=5.0)),
]


def run_backtest(n_seeds=3):
    """Returns {model: {horizon: mean loss}} over all scenarios/seeds."""
    models = {"house": house_baseline, "curve": house_curve_baseline,
              "v1": v1_style, "mine": forecast_case}
    acc = {m: {"final": [], "practice": []} for m in models}
    for name, kw in SCENARIOS:
        for seed in range(n_seeds):
            rng = random.Random(zlib.crc32(("%s/%d" % (name, seed)).encode()))
            for horizon, key in ((108.5, "final"), (1.5, "practice")):
                case, answer = simulate(rng, checkpoint_h=48.0,
                                        horizon_h=horizon, **kw)
                for m, fn in models.items():
                    acc[m][key].append(score_case(answer, fn(case)))
    return {m: {k: sum(v) / len(v) for k, v in d.items()}
            for m, d in acc.items()}


class TestBacktest(unittest.TestCase):
    def test_beats_baselines(self):
        res = run_backtest()
        print("\nmean pinball loss (lower is better):")
        for m, d in res.items():
            print("  %-6s final %.4f  practice %.4f" % (m, d["final"], d["practice"]))
        mine = res["mine"]
        for other in ("house", "curve", "v1"):
            self.assertLess(mine["final"], res[other]["final"],
                            "must beat %s at the final horizon" % other)
            self.assertLess(mine["practice"], res[other]["practice"],
                            "must beat %s at the practice horizon" % other)

    def test_real_practice_reconstruction(self):
        """The real p00 case with its answer reconstructed from the rules'
        description of the first refresh: 638 -> 3,868 players tracked,
        943 -> 4,276 trophies earned, every rate but the 100% one fell.
        New players held ~70% of the old earn rates (that reproduces the
        observed totals within ~1%); the 100% trophy stayed at 100%."""
        import json
        kit = os.path.join(os.path.dirname(__file__), "..", "..",
                           "gta6-trophy-rarity-forecast-01-kit", "input")
        with open(os.path.join(kit, "p00.in")) as fh:
            case = json.load(fh)
        last = case["history"][-1]["rates"]
        n_old, n_new = 638.0, 3868.0
        # New players' earn fraction scales with the old rate: f = 1 for
        # the 100% trophy (every new player had earned it), ~6% for a
        # ~0% trophy. The constant is solved so the totals reproduce the
        # observed 943 -> 4,276 trophies earned exactly.
        s1 = sum(r / 100.0 for r in last)
        s2 = sum((r / 100.0) ** 2 for r in last)
        a = ((4276.0 - 943.0) / (n_new - n_old) - s2) / (s1 - s2)
        answer = []
        for r in last:
            f = a + (1.0 - a) * (r / 100.0)
            e_new = r / 100.0 * n_old + (n_new - n_old) * (r / 100.0) * f
            answer.append(round(100.0 * e_new / n_new, 2))
        total = sum(answer[i] / 100.0 * n_new for i in range(55))
        self.assertAlmostEqual(total, 4276, delta=60)  # sanity: matches
        mine = score_case(answer, forecast_case(case))
        house = score_case(answer, house_baseline(case))
        curve = score_case(answer, house_curve_baseline(case))
        print("\nreconstructed real p00: mine %.4f house %.4f curve %.4f"
              % (mine, house, curve))
        self.assertLess(mine, house)
        self.assertLess(mine, curve)


if __name__ == "__main__":
    unittest.main()
