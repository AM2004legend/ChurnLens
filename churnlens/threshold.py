"""Choose the decision threshold from the business's own costs, not the default 0.5.

For every customer we target with a retention offer we pay `offer_cost`. If they were really
going to churn, the offer saves them with probability `offer_success_rate`, recovering
`customer_value`. So targeting a set T of customers is worth

    savings(T) = Σ_{i ∈ T} (is_churner_i × success_rate × customer_value − offer_cost)

relative to doing nothing. We scan thresholds, target everyone scoring above each one, and keep
the threshold with the largest savings. All costs are assumptions supplied in the YAML.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from churnlens import config  # noqa: E402
from churnlens.io import Costs  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def expected_savings(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                     costs: Costs, value: np.ndarray | float | None = None) -> float:
    """Savings from targeting everyone with score ≥ threshold. `value` may be per customer
    (e.g. from a revenue column); otherwise the assumed `customer_value` is used."""
    y = np.asarray(y_true).astype(float)
    v = costs.customer_value if value is None else np.asarray(value, dtype=float)
    per_customer = y * costs.offer_success_rate * v - costs.offer_cost
    targeted = np.asarray(proba) >= threshold
    return float(np.sum(np.where(targeted, per_customer, 0.0)))


def breakeven_probability(costs: Costs) -> float:
    """If scores were true probabilities, targeting pays when p × s × v > c, i.e. p > c/(s·v).
    Our class-weighted scores are not calibrated, which is why we pick the threshold
    empirically — this value is reported only as a sanity reference."""
    denom = costs.offer_success_rate * costs.customer_value
    return float(costs.offer_cost / denom) if denom else float("inf")


def savings_curve(y_true: np.ndarray, proba: np.ndarray, costs: Costs,
                  value: np.ndarray | float | None = None) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba)
    rows = []
    for t in THRESHOLDS:
        targeted = p >= t
        caught = int((targeted & (y == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "n_targeted": int(targeted.sum()),
                "share_targeted": round(float(targeted.mean()), 4),
                "churners_caught": caught,
                "recall": round(caught / max(int(y.sum()), 1), 4),
                "precision": round(caught / max(int(targeted.sum()), 1), 4),
                "savings": round(expected_savings(y, p, t, costs, value), 2),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame) -> float:
    """Threshold with maximum savings; ties go to the HIGHER threshold (same money, fewer
    offers sent, fewer customers bothered)."""
    best = curve["savings"].max()
    return float(curve.loc[curve["savings"] == best, "threshold"].max())


def compare_strategies(y_true: np.ndarray, proba: np.ndarray, costs: Costs, chosen: float,
                       value: np.ndarray | float | None = None) -> dict[str, dict[str, Any]]:
    """Chosen threshold vs the three obvious alternatives."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba)
    options = {
        "target_no_one": np.inf,
        "target_everyone": -np.inf,
        "threshold_0_5": 0.5,
        "chosen_threshold": chosen,
    }
    out = {}
    for name, t in options.items():
        targeted = p >= t
        out[name] = {
            "threshold": None if np.isinf(t) else float(t),
            "n_targeted": int(targeted.sum()),
            "churners_caught": int((targeted & (y == 1)).sum()),
            "savings": round(expected_savings(y, p, t, costs, value), 2),
        }
    return out


def plot_curve(curve: pd.DataFrame, chosen: float, dataset: str, label: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(curve["threshold"], curve["savings"] / 1e3, color="#4C78A8")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(chosen, color="#E45756", ls="--", lw=1, label=f"chosen {chosen:.2f}")
    ax.axvline(0.5, color="grey", ls=":", lw=1, label="0.5")
    ax.set_xlabel("threshold (target customers with score ≥ threshold)")
    ax.set_ylabel("expected savings (₹ thousands, ASSUMED costs)")
    ax.set_title(f"{dataset}: expected savings vs threshold ({label})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{dataset}_threshold.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path.relative_to(config.ROOT).as_posix())
