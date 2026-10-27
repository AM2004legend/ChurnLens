"""Leakage guard: flag columns that would let a model 'cheat'.

Leakage = information in the features that would not be available at prediction time, or that
is effectively the answer itself. It makes offline metrics look great and real use fail.
The guard only *flags*; a human decides what to drop (recorded in the YAML `drop` list).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from churnlens import config
from churnlens.io import ID, TARGET
from churnlens.schema import CATEGORICAL, DATE_TYPE, NUMERIC

ID_UNIQUE_RATIO = 0.95
AUC_LIMIT = 0.95
SUSPICIOUS_NAME = re.compile(r"churn|cancel|exit|close|terminat", re.IGNORECASE)


@dataclass
class LeakageFlag:
    column: str
    check: str
    value: float | None
    message: str


@dataclass
class LeakageReport:
    flags: list[LeakageFlag] = field(default_factory=list)
    single_feature_auc: dict[str, float] = field(default_factory=dict)

    @property
    def flagged_columns(self) -> list[str]:
        return sorted({f.column for f in self.flags})

    def to_dict(self) -> dict:
        return {
            "flags": [asdict(f) for f in self.flags],
            "single_feature_auc": self.single_feature_auc,
        }


def _looks_discrete(series: pd.Series) -> bool:
    """IDs are integers or strings. Continuous floats (salary, balance) are naturally unique and
    are *not* IDs, so they are excluded from the unique-ratio check."""
    if pd.api.types.is_float_dtype(series):
        s = series.dropna()
        return bool(len(s)) and bool((s == s.round()).all())
    return True


def check_id_like(df: pd.DataFrame, types: dict[str, str]) -> list[LeakageFlag]:
    flags = []
    n = len(df)
    for col, kind in types.items():
        if kind == DATE_TYPE or not n:
            continue
        ratio = df[col].nunique(dropna=True) / n
        if ratio > ID_UNIQUE_RATIO and _looks_discrete(df[col]):
            flags.append(
                LeakageFlag(
                    col, "id_like", round(ratio, 4),
                    f"{ratio:.0%} unique values — looks like an identifier; a model can memorise "
                    "it but it carries no general signal.",
                )
            )
    return flags


def _oof_target_encode(x: pd.Series, y: pd.Series, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold churn rate per category.

    In-sample encoding would give a near-perfect AUC to any high-cardinality column (each
    surname 'predicts' its own row), so every row is encoded using the other folds only.
    """
    x = x.astype("string").fillna("<missing>")
    encoded = np.full(len(x), float(y.mean()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    for fit_idx, enc_idx in skf.split(x, y):
        rates = y.iloc[fit_idx].groupby(x.iloc[fit_idx].values).mean()
        mapped = x.iloc[enc_idx].map(rates).astype(float)
        encoded[enc_idx] = mapped.fillna(float(y.iloc[fit_idx].mean())).to_numpy()
    return encoded


def single_feature_aucs(df: pd.DataFrame, types: dict[str, str]) -> dict[str, float]:
    """How well each column alone separates churners from stayers (0.5 = useless, 1 = perfect).

    Direction does not matter, so max(AUC, 1 − AUC) is reported.
    """
    y = df[TARGET].astype(int)
    out: dict[str, float] = {}
    if y.nunique() < 2:
        return out
    for col, kind in types.items():
        if kind == NUMERIC:
            x = pd.to_numeric(df[col], errors="coerce").astype(float)
            x = x.fillna(x.median()).to_numpy()
        elif kind == CATEGORICAL:
            x = _oof_target_encode(df[col], y)
        else:
            continue
        if np.nanstd(x) == 0:
            out[col] = 0.5
            continue
        auc = roc_auc_score(y, x)
        out[col] = round(float(max(auc, 1 - auc)), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def check_leakage(df: pd.DataFrame, types: dict[str, str]) -> LeakageReport:
    """Run all checks. `df` should be the TRAINING split only."""
    report = LeakageReport()
    report.flags += check_id_like(df, types)

    report.single_feature_auc = single_feature_aucs(df, types)
    for col, auc in report.single_feature_auc.items():
        if auc > AUC_LIMIT:
            report.flags.append(
                LeakageFlag(
                    col, "high_single_auc", auc,
                    f"This column alone separates churners with AUC {auc:.3f} — real predictors "
                    "are rarely this strong; check it is not recorded after the churn event.",
                )
            )

    for col in types:
        if col not in (ID, TARGET) and SUSPICIOUS_NAME.search(col):
            report.flags.append(
                LeakageFlag(
                    col, "suspicious_name", None,
                    "Name mentions churn/cancel/exit/close/terminate — may describe the outcome.",
                )
            )

    for col, kind in types.items():
        if kind == DATE_TYPE:
            report.flags.append(
                LeakageFlag(
                    col, "date_column", None,
                    "Date column — confirm it is known BEFORE the churn outcome (e.g. a "
                    "'last login' date taken after cancelling would leak the answer).",
                )
            )
    return report


def to_markdown(report: LeakageReport, dataset: str, dropped: list[str]) -> str:
    lines = [f"# Leakage report — {dataset}", "", "Computed on the training split only.", ""]
    if report.flags:
        lines += ["| Column | Check | Value | Dropped in config? | Why flagged |",
                  "|---|---|---|---|---|"]
        for f in report.flags:
            value = "" if f.value is None else f"{f.value}"
            drop = "yes" if f.column in dropped else "**no**"
            lines.append(f"| {f.column} | {f.check} | {value} | {drop} | {f.message} |")
    else:
        lines.append("No columns flagged.")
    lines += ["", "## Single-feature ROC-AUC (max(AUC, 1−AUC))", "",
              "| Column | AUC |", "|---|---|"]
    lines += [f"| {c} | {a} |" for c, a in report.single_feature_auc.items()]
    return "\n".join(lines) + "\n"


def save_report(report: LeakageReport, dataset: str, dropped: list[str]) -> None:
    out_dir = config.REPORTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict() | {"dropped_in_config": dropped}
    (out_dir / "leakage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "leakage.md").write_text(to_markdown(report, dataset, dropped), encoding="utf-8")
