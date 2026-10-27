"""Hand-calculated example.

4 customers, offer ₹100, value ₹1000, success 50% → saving a churner is worth 0.5×1000 = ₹500.
Targeting a churner: +500 − 100 = +400.  Targeting a stayer: −100.

  scores  [0.9, 0.8, 0.3, 0.1]
  churner [ 1,   0,   1,   0 ]

  t = 0.5  → target #1, #2          → 400 − 100           = 300
  t = 0.2  → target #1, #2, #3      → 400 − 100 + 400     = 700   ← best
  t = 0.05 → target everyone        → 700 − 100           = 600
  nobody                            → 0
Every threshold in (0.1, 0.3] gives 700; the tie goes to the highest, 0.30.
"""

import numpy as np
import pytest

from churnlens import threshold as th
from churnlens.io import Costs

Y = np.array([1, 0, 1, 0])
P = np.array([0.9, 0.8, 0.3, 0.1])
COSTS = Costs(offer_cost=100, customer_value=1000, offer_success_rate=0.5)


@pytest.mark.parametrize("t, expected", [(0.5, 300), (0.2, 700), (0.05, 600), (0.95, 0)])
def test_expected_savings_hand_calculated(t: float, expected: float) -> None:
    assert th.expected_savings(Y, P, t, COSTS) == expected


def test_choose_threshold_picks_max_and_breaks_ties_high() -> None:
    curve = th.savings_curve(Y, P, COSTS)
    assert curve["savings"].max() == 700
    assert th.choose_threshold(curve) == 0.30


def test_compare_strategies() -> None:
    s = th.compare_strategies(Y, P, COSTS, chosen=0.30)
    assert s["target_no_one"]["savings"] == 0
    assert s["target_everyone"]["savings"] == 600
    assert s["threshold_0_5"]["savings"] == 300
    assert s["chosen_threshold"]["savings"] == 700
    assert s["chosen_threshold"]["churners_caught"] == 2


def test_per_customer_value() -> None:
    # Only the first churner is valuable: 0.5×2000 − 100 = 900; second: 0.5×0 − 100 = −100.
    v = np.array([2000, 0, 0, 0])
    assert th.expected_savings(Y, P, 0.2, COSTS, value=v) == 900 - 100 - 100


def test_breakeven_probability() -> None:
    assert th.breakeven_probability(COSTS) == pytest.approx(0.2)


def test_curve_covers_005_to_095() -> None:
    curve = th.savings_curve(Y, P, COSTS)
    assert curve["threshold"].iloc[0] == 0.05 and curve["threshold"].iloc[-1] == 0.95
