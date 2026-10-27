"""End-to-end pipeline and CLI.

    python -m churnlens.run --dataset bank
    python -m churnlens.run --all

The pipeline is split into `prepare` (load → clean → split → leakage → EDA) and `train_and_evaluate`
(CV → pick model → fit → test) so the Streamlit app can show leakage warnings and let the user
choose drops in between, using exactly the same functions as the CLI.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from churnlens import config, eda, features, io, leakage, models, schema, transactions
from churnlens.io import ID, TARGET, DatasetConfig
from churnlens.schema import CATEGORICAL, NUMERIC
from churnlens.utils import md_table, write_json


@dataclass
class Prepared:
    cfg: DatasetConfig
    train: pd.DataFrame
    test: pd.DataFrame
    types: dict[str, str]
    split_info: dict[str, Any]
    cleaning_log: dict[str, Any]
    leakage: leakage.LeakageReport
    eda: eda.EdaResult


@dataclass
class Trained:
    numeric: list[str]
    categorical: list[str]
    cv_table: pd.DataFrame
    best: str
    oof: dict[str, np.ndarray]
    model: Any
    test_proba: np.ndarray
    test_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)


def prepare(cfg: DatasetConfig, raw: pd.DataFrame | None = None,
            save_figures: bool = True) -> Prepared:
    """Load, clean and split; then run the leakage guard and EDA on the TRAIN split only."""
    raw = io.read_raw(cfg) if raw is None else raw
    mapped = io.apply_mapping(raw, cfg)
    clean, log = schema.clean(cfg, mapped)

    if cfg.mode == "labelled":
        types = schema.infer_types(clean)
        groups = None
        if cfg.group_by:
            clean[features.GROUP] = features.group_key(clean, cfg.group_by)
            groups = clean[features.GROUP]
        train, test = features.split_labelled(clean, groups=groups)
        method = "grouped stratified" if cfg.group_by else "stratified random"
        split_info = {
            "method": f"{method} {1 - features.TEST_SIZE:.0%}/{features.TEST_SIZE:.0%}",
            "group_by": cfg.group_by,
            "train_rows": len(train), "test_rows": len(test),
            "train_churn_rate": round(float(train[TARGET].mean()), 4),
            "test_churn_rate": round(float(test[TARGET].mean()), 4),
        }
    else:
        train, test, split_info = transactions.time_split(
            clean, cfg.inactivity_days, cfg.lookback_days
        )
        split_info = {"method": "time-based (train cutoff < test cutoff)", **split_info}
        types = {c: NUMERIC for c in transactions.FEATURES}

    report = leakage.check_leakage(train, types)
    eda_result = eda.run_eda(train, types, cfg.name, save_figures=save_figures)
    return Prepared(cfg, train, test, types, split_info, log, report, eda_result)


def feature_lists(types: dict[str, str], drop: list[str]) -> tuple[list[str], list[str]]:
    """Numeric and categorical model inputs. Date columns are not used as raw features."""
    keep = {c: k for c, k in types.items() if c not in set(drop) and c not in (ID, TARGET)}
    return ([c for c, k in keep.items() if k == NUMERIC],
            [c for c, k in keep.items() if k == CATEGORICAL])


def train_and_evaluate(prep: Prepared, drop: list[str] | None = None) -> Trained:
    """CV both models on train, pick the best by CV PR-AUC, refit on all of train, and score
    the test set once."""
    drop = prep.cfg.drop if drop is None else drop
    numeric, categorical = feature_lists(prep.types, drop)
    cols = numeric + categorical
    X, y = prep.train[cols], prep.train[TARGET]
    groups = prep.train[features.GROUP] if features.GROUP in prep.train else None
    cv_table, oof = models.cross_validate(X, y, numeric, categorical, groups=groups)
    best = models.choose_best(cv_table)

    test_metrics, fitted = {}, {}
    for name in models.MODEL_NAMES:
        fitted[name] = models.fit(name, X, y, numeric, categorical)
        proba = fitted[name].predict_proba(prep.test[cols])[:, 1]
        test_metrics[name] = models.evaluate(prep.test[TARGET], proba, threshold=0.5)
    test_proba = fitted[best].predict_proba(prep.test[cols])[:, 1]
    return Trained(numeric, categorical, cv_table, best, oof, fitted[best], test_proba,
                   test_metrics)


def save_reports(prep: Prepared, trained: Trained) -> dict[str, Any]:
    name = prep.cfg.name
    out_dir = config.REPORTS_DIR / name
    leakage.save_report(prep.leakage, name, prep.cfg.drop, bool(prep.cfg.group_by))
    eda.save_eda(prep.eda)
    schema.save_cleaning_report(name, schema.missing_report(prep.train, prep.types),
                                prep.cleaning_log)
    payload = {
        "dataset": name,
        "mode": prep.cfg.mode,
        "split": prep.split_info,
        "features": {"numeric": trained.numeric, "categorical": trained.categorical,
                     "dropped": prep.cfg.drop},
        "cv": trained.cv_table.to_dict(orient="records"),
        "selection_metric": models.SELECTION_METRIC,
        "best_model": trained.best,
        "test_at_0_5": trained.test_metrics,
    }
    write_json(out_dir / "metrics.json", payload)
    return payload


def run_dataset(name: str, save_figures: bool = True) -> dict[str, Any]:
    t0 = time.time()
    cfg = io.load_config(name)
    prep = prepare(cfg, save_figures=save_figures)
    undropped = leakage.unresolved(prep.leakage, cfg.drop, bool(cfg.group_by))
    if undropped:
        print(f"[{name}] WARNING leakage flags not dropped in config: {undropped}")
    trained = train_and_evaluate(prep)
    models.save_model(trained.model, name, trained.best)
    payload = save_reports(prep, trained)
    payload["seconds"] = round(time.time() - t0, 1)
    return payload


def _print_summary(p: dict[str, Any]) -> None:
    print(f"\n=== {p['dataset']} ({p['mode']}) — {p['seconds']}s")
    print("split:", p["split"])
    cv = pd.DataFrame(p["cv"])[["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean",
                                "pr_auc_std", "recall_mean", "precision_mean"]]
    print("CV on train:\n" + md_table(cv))
    print(f"best by CV {p['selection_metric']}: {p['best_model']}")
    rows = [{"model": m, **{k: v for k, v in r.items() if k != "confusion_matrix"},
             **r["confusion_matrix"]} for m, r in p["test_at_0_5"].items()]
    print("TEST @0.5:\n" + md_table(pd.DataFrame(rows)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the ChurnLens pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=io.available_datasets())
    group.add_argument("--all", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    names = io.available_datasets() if args.all else [args.dataset]
    for name in names:
        _print_summary(run_dataset(name, save_figures=not args.no_figures))


if __name__ == "__main__":
    main()
