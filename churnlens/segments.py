"""Group at-risk customers by their main churn driver and attach a suggested action.

A list of 400 at-risk IDs is hard to act on; "120 inactive members → re-engagement campaign,
80 single-product customers → bundle offer" is a plan. Segment names and actions come from a
rules table in each dataset's YAML (`segments:`), so the business owns the wording.
"""

from __future__ import annotations

import pandas as pd

from churnlens import config
from churnlens.io import ID

DEFAULT_ACTION = "Review the account and make a personalised retention offer"


def rule_for(driver: str | None, rules: dict[str, dict[str, str]]) -> tuple[str, str]:
    if driver is None or pd.isna(driver):  # pandas stores a missing driver as NaN
        return "No clear driver", DEFAULT_ACTION
    rule = rules.get(driver)
    if rule:
        return rule.get("name", f"Driven by {driver}"), rule.get("action", DEFAULT_ACTION)
    return f"Driven by {driver.replace('_', ' ')}", DEFAULT_ACTION


def assign_segments(explained: pd.DataFrame, threshold: float,
                    rules: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Keep customers at or above the chosen threshold and label each by its top driver."""
    at_risk = explained[explained["churn_score"] >= threshold].copy()
    named = [rule_for(d, rules) for d in at_risk["top_driver"]]
    at_risk["segment"] = [n for n, _ in named]
    at_risk["suggested_action"] = [a for _, a in named]
    return at_risk.sort_values("churn_score", ascending=False).reset_index(drop=True)


def segment_summary(segmented: pd.DataFrame) -> pd.DataFrame:
    if segmented.empty:
        return pd.DataFrame(columns=["segment", "top_driver", "suggested_action", "customers",
                                     "share", "avg_score"])
    g = segmented.groupby(["segment", "suggested_action"], dropna=False)
    out = g.agg(top_driver=("top_driver", "first"), customers=(ID, "size"),
                avg_score=("churn_score", "mean")).reset_index()
    out["share"] = (out["customers"] / out["customers"].sum()).round(4)
    out["avg_score"] = out["avg_score"].round(3)
    cols = ["segment", "top_driver", "suggested_action", "customers", "share", "avg_score"]
    return out.sort_values("customers", ascending=False)[cols].reset_index(drop=True)


def targets_csv(segmented: pd.DataFrame) -> pd.DataFrame:
    """The downloadable target list: IDs, segment and action only — no customer attributes."""
    return segmented[[ID, "segment", "suggested_action"]]


def save_targets(segmented: pd.DataFrame, dataset: str) -> str:
    """Written under a `*targets*` name, which .gitignore excludes (it lists real IDs)."""
    path = config.REPORTS_DIR / dataset / f"{dataset}_targets.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    targets_csv(segmented).to_csv(path, index=False)
    return str(path.relative_to(config.ROOT).as_posix())
