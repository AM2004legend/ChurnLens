"""Column typing, missing-value report and cleaning.

Cleaning here is deliberately *row-level and rule-based* (fix labels, drop broken rows). Anything
that learns from the data — imputing medians, encoding categories, scaling — happens later inside
a scikit-learn Pipeline fitted on the training split only, so it cannot leak test information.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from churnlens import config
from churnlens.io import AMOUNT, DATE, ID, INVOICE, TARGET, DatasetConfig

NUMERIC, CATEGORICAL, DATE_TYPE = "numeric", "categorical", "date"

_DATE_PATTERN = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
_PARSE_SHARE = 0.95  # share of non-null values that must parse to accept a type


def infer_column_type(series: pd.Series) -> str:
    """Classify one column as numeric, categorical or date.

    Text columns that are almost entirely numbers ("1,200" style aside) or dates are promoted,
    because CSVs often store them as strings.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATE_TYPE
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return NUMERIC
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return CATEGORICAL
    if pd.to_numeric(values, errors="coerce").notna().mean() >= _PARSE_SHARE:
        return NUMERIC
    if values.str.match(_DATE_PATTERN).mean() >= _PARSE_SHARE:
        return DATE_TYPE
    return CATEGORICAL


def infer_types(df: pd.DataFrame, exclude: tuple[str, ...] = (ID, TARGET)) -> dict[str, str]:
    """Type every feature column (the id and label are excluded — they are not features)."""
    return {c: infer_column_type(df[c]) for c in df.columns if c not in exclude}


def missing_report(df: pd.DataFrame, types: dict[str, str] | None = None) -> pd.DataFrame:
    """Missing count and % per column, worst first. Empty strings count as missing."""
    blank = df.apply(lambda s: s.astype("string").str.strip().eq("").fillna(False))
    n_missing = df.isna().sum() + blank.sum()
    report = pd.DataFrame(
        {
            "column": df.columns,
            "type": [(types or {}).get(c, "") for c in df.columns],
            "n_missing": n_missing.values.astype(int),
            "pct_missing": (n_missing.values / max(len(df), 1) * 100).round(2),
        }
    )
    return report.sort_values(["n_missing", "column"], ascending=[False, True]).reset_index(
        drop=True
    )


def _normalise_ids(ids: pd.Series) -> pd.Series:
    """Turn ids like 13085.0 into '13085' so the same customer always has one key."""
    if pd.api.types.is_float_dtype(ids):
        whole = ids.dropna()
        if (whole == whole.round()).all():
            return ids.astype("Int64").astype("string")
    return ids.astype("string").str.strip()


def clean_labelled(
    df: pd.DataFrame, cfg: DatasetConfig, types: dict[str, str] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a mapped labelled table. Returns the cleaned frame and a log of what changed.

    Missing feature values are kept (the model Pipeline imputes them from training data);
    only rows without a label or id are dropped, since they cannot be used at all.
    """
    types = types or infer_types(df)
    out = df.copy()
    log: dict[str, Any] = {"rows_in": len(out)}

    for col, kind in types.items():
        if kind == CATEGORICAL:
            s = out[col].astype("string").str.strip()
            s = s.mask(s == "")
            mapping = cfg.category_map.get(col, {})
            if mapping:
                s = s.replace({str(k): str(v) for k, v in mapping.items()})
            # object dtype with np.nan is what sklearn imputers/encoders handle most reliably.
            out[col] = s.astype(object).where(s.notna(), None)
        elif kind == NUMERIC and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif kind == DATE_TYPE:
            out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
    log["category_maps_applied"] = sorted(c for c in cfg.category_map if c in types)

    missing_label = out[TARGET].isna() | out[ID].isna()
    log["dropped_missing_id_or_label"] = int(missing_label.sum())
    out = out[~missing_label]

    dup_rows = out.duplicated()
    log["dropped_duplicate_rows"] = int(dup_rows.sum())
    out = out[~dup_rows]

    dup_ids = out[ID].duplicated(keep="first")
    log["dropped_duplicate_ids"] = int(dup_ids.sum())
    out = out[~dup_ids]

    out[ID] = _normalise_ids(out[ID])
    out[TARGET] = out[TARGET].astype(int)
    log["rows_out"] = len(out)
    log["churn_rate"] = round(float(out[TARGET].mean()), 4)
    return out.reset_index(drop=True), log


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a mapped transaction log.

    Rows without a customer cannot be attributed, and non-positive amounts are returns or
    cancellations — counting them as purchases would make a customer look more active.
    Exact duplicate rows are dropped (in Online Retail II the two yearly sheets overlap).
    """
    out = df.copy()
    log: dict[str, Any] = {"rows_in": len(out)}

    no_customer = out[ID].isna()
    log["dropped_missing_customer"] = int(no_customer.sum())
    out = out[~no_customer]

    bad = out[DATE].isna() | out[AMOUNT].isna()
    log["dropped_missing_date_or_amount"] = int(bad.sum())
    out = out[~bad]

    dups = out.duplicated()
    log["dropped_duplicate_rows"] = int(dups.sum())
    out = out[~dups]

    non_positive = out[AMOUNT] <= 0
    if INVOICE in out.columns:
        non_positive |= out[INVOICE].str.upper().str.startswith("C").fillna(False)
    log["dropped_returns_or_zero_amount"] = int(non_positive.sum())
    out = out[~non_positive]

    # Raw passthrough columns were only needed to judge duplicates.
    out = out[[c for c in (ID, DATE, AMOUNT, INVOICE) if c in out.columns]]
    out[ID] = _normalise_ids(out[ID])
    out = out.sort_values([ID, DATE]).reset_index(drop=True)
    log["rows_out"] = len(out)
    log["customers"] = int(out[ID].nunique())
    log["date_min"] = str(out[DATE].min())
    log["date_max"] = str(out[DATE].max())
    return out, log


def clean(cfg: DatasetConfig, mapped: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean according to the dataset's mode."""
    if cfg.mode == "labelled":
        return clean_labelled(mapped, cfg)
    return clean_transactions(mapped)


def save_processed(df: pd.DataFrame, name: str) -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_pickle(config.DATA_PROCESSED / f"{name}_clean.pkl")


def save_cleaning_report(name: str, missing: pd.DataFrame, log: dict[str, Any]) -> None:
    """Aggregates only (counts, %), never customer rows — safe to commit."""
    out_dir = config.REPORTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"cleaning": log, "missing": missing.to_dict(orient="records")}
    (out_dir / "cleaning.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
