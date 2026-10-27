"""Transactions mode: build a churn label and RFM features from a raw purchase log.

Most businesses have no "churned" column — only purchases. We define churn ourselves:
pick a cutoff date c, describe each customer using ONLY purchases on or before c, and call them
churned if they make no purchase in the next N days (c, c + N]. Everything is at day level.

Because the label is defined by time, train and test must be separated by time too: the model is
trained at an earlier cutoff and judged at a later one, exactly as it would be used.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from churnlens.io import AMOUNT, DATE, ID, INVOICE, TARGET

TREND_DAYS = 90

FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "tenure_days",
    "purchases_last_90",
    "purchases_prev_90",
    "purchase_trend",
]


def default_cutoffs(last_date: pd.Timestamp, inactivity_days: int) -> tuple[pd.Timestamp, ...]:
    """Test cutoff c2 = last − N (so its label window is fully observed);
    train cutoff c1 = c2 − N = last − 2N (so train labels end exactly where test begins)."""
    last = pd.Timestamp(last_date).normalize()
    c2 = last - pd.Timedelta(days=inactivity_days)
    c1 = c2 - pd.Timedelta(days=inactivity_days)
    return c1, c2


def _purchases(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per purchase. With an invoice column, invoice lines are summed into one order;
    otherwise each row already is a purchase."""
    df = tx.assign(day=tx[DATE].dt.normalize())
    if INVOICE in df.columns:
        return (
            df.groupby([ID, INVOICE], as_index=False, observed=True)
            .agg(day=("day", "min"), amount=(AMOUNT, "sum"))
        )
    return df[[ID, "day", AMOUNT]].rename(columns={AMOUNT: "amount"})


def build_snapshot(
    tx: pd.DataFrame,
    cutoff: pd.Timestamp,
    inactivity_days: int = 90,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Customer-level features at `cutoff` plus the churn label from the following N days.

    Raises if the data does not extend to cutoff + N: customers would then look churned only
    because we cannot see their future purchases (a silent, very damaging labelling bug).
    """
    cutoff = pd.Timestamp(cutoff).normalize()
    p = _purchases(tx)
    label_end = cutoff + pd.Timedelta(days=inactivity_days)
    if p["day"].max() < label_end:
        raise ValueError(
            f"label window (c, c+{inactivity_days}d] ends {label_end.date()} but data ends "
            f"{p['day'].max().date()} — choose an earlier cutoff or a shorter window"
        )

    hist = p[p["day"] <= cutoff]
    look_start = cutoff - pd.Timedelta(days=lookback_days)
    look = hist[hist["day"] > look_start]
    active = look[ID].unique()
    hist = hist[hist[ID].isin(active)]

    g_all = hist.groupby(ID)
    g_look = look.groupby(ID)
    out = pd.DataFrame(index=pd.Index(active, name=ID))
    out["recency_days"] = (cutoff - g_all["day"].max()).dt.days
    out["frequency"] = g_look.size()
    out["monetary_total"] = g_look["amount"].sum()
    out["monetary_avg"] = out["monetary_total"] / out["frequency"]
    out["tenure_days"] = (cutoff - g_all["day"].min()).dt.days

    last_start = cutoff - pd.Timedelta(days=TREND_DAYS)
    prev_start = last_start - pd.Timedelta(days=TREND_DAYS)
    in_last = hist[hist["day"] > last_start]
    in_prev = hist[(hist["day"] > prev_start) & (hist["day"] <= last_start)]
    out["purchases_last_90"] = in_last.groupby(ID).size().reindex(out.index, fill_value=0)
    out["purchases_prev_90"] = in_prev.groupby(ID).size().reindex(out.index, fill_value=0)
    out["purchase_trend"] = out["purchases_last_90"] - out["purchases_prev_90"]

    future = p[(p["day"] > cutoff) & (p["day"] <= label_end)]
    out[TARGET] = (~out.index.isin(future[ID].unique())).astype(int)

    out = out.reset_index().sort_values(ID).reset_index(drop=True)
    int_cols = ["recency_days", "frequency", "tenure_days", "purchases_last_90",
                "purchases_prev_90", "purchase_trend"]
    out[int_cols] = out[int_cols].astype(int)
    return out[[ID, TARGET, *FEATURES]]


def time_split(
    tx: pd.DataFrame,
    inactivity_days: int = 90,
    lookback_days: int = 365,
    cutoffs: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train snapshot at c1, test snapshot at c2 > c1.

    The same customer can appear in both — that is realistic (we re-score customers over
    time); what matters is that every test label lies in the future of everything trained on.
    """
    c1, c2 = cutoffs or default_cutoffs(tx[DATE].max(), inactivity_days)
    if not c1 < c2:
        raise ValueError(f"train cutoff {c1} must be before test cutoff {c2}")
    train = build_snapshot(tx, c1, inactivity_days, lookback_days)
    test = build_snapshot(tx, c2, inactivity_days, lookback_days)
    info = {
        "train_cutoff": str(pd.Timestamp(c1).date()),
        "test_cutoff": str(pd.Timestamp(c2).date()),
        "last_date": str(tx[DATE].max().normalize().date()),
        "inactivity_days": inactivity_days,
        "lookback_days": lookback_days,
        "train_customers": len(train),
        "test_customers": len(test),
        "train_churn_rate": round(float(train[TARGET].mean()), 4),
        "test_churn_rate": round(float(test[TARGET].mean()), 4),
    }
    return train, test, info
