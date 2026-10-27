"""Synthetic demo data, so the app and CI work without the Kaggle/UCI files.

The raw datasets are gitignored (licences, size), which means a deployed app has none of them.
These generators create small, obviously fake datasets with planted, known churn drivers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnlens import io


def labelled(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Telecom-style customer table. Planted drivers: older age, more support calls, basic
    plan, short tenure."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    plan = rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2])
    tenure = rng.integers(0, 72, n)
    calls = rng.poisson(2, n).astype(float)
    calls[rng.random(n) < 0.04] = np.nan
    monthly = np.round(rng.normal(40, 12, n).clip(10), 2)
    logit = (0.04 * (age - 45) + 0.35 * np.nan_to_num(calls, nan=2) + (plan == "basic") * 0.9
             - 0.03 * tenure - 0.9)
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "CustID": np.arange(100_000, 100_000 + n),
        "Status": np.where(churned, "Closed", "Open"),
        "Age": age,
        "Plan": plan,
        "TenureMonths": tenure,
        "SupportCalls": calls,
        "MonthlyCharge": monthly,
    })


def labelled_config() -> io.DatasetConfig:
    return io.parse_config({
        "name": "synthetic_telecom", "mode": "labelled", "path": "synthetic.csv",
        "columns": {"customer_id": "CustID", "churn_label": "Status",
                    "churn_positive_value": "Closed"},
        "costs": {"offer_cost": 300, "customer_value": 3000, "offer_success_rate": 0.3},
        "segments": {
            "SupportCalls": {"name": "Frequent support callers",
                             "action": "Proactive support call"},
            "TenureMonths": {"name": "New customers", "action": "Onboarding check-in"},
            "Plan": {"name": "Basic-plan customers", "action": "Upgrade trial offer"},
            "Age": {"name": "Older customers", "action": "Simplified plan and priority line"},
        },
    })


def transactions(n_customers: int = 300, days: int = 760, seed: int = 8) -> pd.DataFrame:
    """Purchase log where each customer buys at their own rate and may stop at some point."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    frames = []
    for c in range(n_customers):
        rate = rng.uniform(0.005, 0.08)
        stop = rng.integers(200, days)
        d = np.flatnonzero(rng.random(days) < rate)
        d = d[d < stop]
        if not len(d):
            continue
        frames.append(pd.DataFrame({
            "cust": c,
            "ts": start + pd.to_timedelta(d, unit="D"),
            "qty": rng.integers(1, 5, len(d)),
            "price": rng.uniform(2, 50, len(d)).round(2),
            "inv": [f"I{c}-{x}" for x in d],
        }))
    return pd.concat(frames, ignore_index=True)


def transactions_config() -> io.DatasetConfig:
    return io.parse_config({
        "name": "synthetic_shop", "mode": "transactions", "path": "synthetic.csv",
        "columns": {"customer_id": "cust", "date": "ts", "amount": ["qty", "price"],
                    "invoice": "inv"},
        "transactions": {"inactivity_days": 60, "lookback_days": 365},
        "costs": {"offer_cost": 200, "customer_value": 2500, "offer_success_rate": 0.3},
    })
