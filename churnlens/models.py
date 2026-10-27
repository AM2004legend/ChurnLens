"""Train, cross-validate, evaluate and save models.

Two models on purpose: a scaled logistic regression is the honest baseline (fast, explainable);
LightGBM captures non-linear effects and interactions. If LightGBM can't beat the baseline,
that is a finding, not a failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from churnlens import config
from churnlens.features import build_preprocessor

MODEL_NAMES = ("logreg", "lightgbm")
CV_FOLDS = 5
SELECTION_METRIC = "pr_auc"


def make_model(name: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Preprocessing + classifier in one Pipeline, so CV refits preprocessing inside each fold.

    Both use balanced class weights: churners are the minority, and without weighting a model
    can score well by mostly predicting "stays".
    """
    if name == "logreg":
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=config.RANDOM_STATE
        )
        return Pipeline([("prep", build_preprocessor(numeric, categorical, scale=True)),
                         ("clf", clf)])
    if name == "lightgbm":
        clf = LGBMClassifier(
            class_weight="balanced",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        return Pipeline([("prep", build_preprocessor(numeric, categorical, scale=False)),
                         ("clf", clf)])
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """All the metrics we always report together. Accuracy is deliberately absent: with 20%
    churners, predicting "nobody churns" is 80% accurate and useless."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "churn_rate": round(float(y_true.mean()), 4),
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "share_targeted": round(float(pred.mean()), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def cross_validate(
    X: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str],
    names: tuple[str, ...] = MODEL_NAMES, groups: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """5-fold stratified CV on TRAINING data only.

    Returns a mean ± std table and the out-of-fold probabilities: every training row gets a
    prediction from a model that never saw it. Those OOF scores are what the threshold step
    tunes on, so the test set stays untouched until the very end. With `groups`, copies of a
    customer stay in the same fold (same reason as the grouped train/test split).
    """
    y_arr = y.to_numpy().astype(int)
    if groups is None:
        folds = list(StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                     random_state=config.RANDOM_STATE).split(X, y_arr))
    else:
        folds = list(StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True,
                                          random_state=config.RANDOM_STATE)
                     .split(X, y_arr, groups))
    rows, oof = [], {}
    for name in names:
        template = make_model(name, numeric, categorical)
        scores = np.zeros(len(y_arr))
        fold_metrics = []
        for fit_idx, val_idx in folds:
            model = clone(template).fit(X.iloc[fit_idx], y_arr[fit_idx])
            p = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores[val_idx] = p
            fold_metrics.append(evaluate(y_arr[val_idx], p))
        oof[name] = scores
        row: dict[str, Any] = {"model": name}
        for m in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
            vals = [f[m] for f in fold_metrics]
            row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{m}_std"] = round(float(np.std(vals)), 4)
        rows.append(row)
    return pd.DataFrame(rows), oof


def choose_best(cv_table: pd.DataFrame, metric: str = SELECTION_METRIC) -> str:
    """Pick by CV PR-AUC: it focuses on how well churners are ranked at the top, which is what
    a retention campaign uses. ROC-AUC can look good even when the top of the list is noisy."""
    return str(cv_table.sort_values(f"{metric}_mean", ascending=False).iloc[0]["model"])


def fit(name: str, X: pd.DataFrame, y: pd.Series, numeric: list[str],
        categorical: list[str]) -> Pipeline:
    return make_model(name, numeric, categorical).fit(X, y.astype(int))


def save_model(model: Pipeline, dataset: str, name: str) -> Path:
    """Models go to `models/` (gitignored) — they can embed category values from the data."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{dataset}_{name}.joblib"
    joblib.dump(model, path)
    return path
