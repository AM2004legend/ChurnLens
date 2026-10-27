"""SHAP explanations: which features push each customer's churn score up or down.

SHAP splits a single prediction into additive contributions, one per feature, measured against
an average customer. Summing |contribution| over customers gives global importance; sorting one
customer's contributions gives their personal "top reasons". Both models are explained in
log-odds units. These show what the MODEL uses — not proven causes.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from churnlens import config  # noqa: E402
from churnlens.features import source_column  # noqa: E402
from churnlens.io import ID  # noqa: E402

BACKGROUND_ROWS = 1000


def shap_matrix(model: Pipeline, X: pd.DataFrame, background: pd.DataFrame) -> tuple[
        np.ndarray, list[str]]:
    """SHAP values for the positive (churn) class on the transformed features."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = np.asarray(prep.transform(X), dtype=float)
    names = list(prep.get_feature_names_out())
    if isinstance(clf, LGBMClassifier):
        with warnings.catch_warnings():
            # shap warns that binary LightGBM output is now a per-class list; handled below.
            warnings.filterwarnings("ignore", message=".*has changed to a list of ndarray.*")
            values = shap.TreeExplainer(clf).shap_values(Xt)
    elif isinstance(clf, LogisticRegression):
        bg = background.sample(min(len(background), BACKGROUND_ROWS),
                               random_state=config.RANDOM_STATE)
        masker = np.asarray(prep.transform(bg), dtype=float)
        values = shap.LinearExplainer(clf, masker).shap_values(Xt)
    else:
        raise TypeError(f"No SHAP explainer configured for {type(clf).__name__}")
    # Different shap versions return a list per class or an (n, k, 2) array.
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values, names


def to_columns(values: np.ndarray, names: list[str], numeric: list[str],
               categorical: list[str]) -> pd.DataFrame:
    """Sum one-hot pieces and missing-indicators back into their business column. Because SHAP
    values are additive, the sum is the column's total contribution."""
    sources = [source_column(n, numeric, categorical) for n in names]
    frame = pd.DataFrame(values, columns=names)
    return frame.T.groupby(sources, sort=False).sum().T


def global_importance(contrib: pd.DataFrame) -> pd.Series:
    return contrib.abs().mean().sort_values(ascending=False).round(4)


def plot_global(importance: pd.Series, dataset: str, top: int = 12) -> str:
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_xlabel("mean |SHAP| (log-odds) on test customers")
    ax.set_title(f"{dataset}: what the model relies on most", fontsize=10)
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{dataset}_shap_global.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path.relative_to(config.ROOT).as_posix())


def _label(col: str) -> str:
    return col.replace("_", " ")


def _fmt(v: Any) -> str:
    if isinstance(v, int | float | np.integer | np.floating):
        v = float(v)
        return f"{v:,.0f}" if abs(v) >= 100 or v == int(v) else f"{v:.2f}"
    return str(v)


def reason_text(col: str, value: Any, typical: Any) -> str:
    """One plain-English reason, e.g. 'Age is 58 (typical customer: 37)'."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return f"{_label(col)} is not recorded"
    if typical is not None and isinstance(value, int | float | np.integer | np.floating):
        return f"{_label(col)} is {_fmt(value)} (typical customer: {_fmt(typical)})"
    return f"{_label(col)} is '{value}'"


def typical_values(train: pd.DataFrame, numeric: list[str]) -> dict[str, float]:
    """Training medians: the 'typical customer' a reason is compared against."""
    return {c: float(pd.to_numeric(train[c], errors="coerce").median()) for c in numeric}


def explain_customers(contrib: pd.DataFrame, X: pd.DataFrame, ids: pd.Series,
                      scores: np.ndarray, typical: dict[str, float], k: int = 3) -> pd.DataFrame:
    """Top-k risk-RAISING reasons per customer (positive SHAP), in plain words.

    Only positive contributions are used: a retention team wants to know why someone is at
    risk, not what keeps them safe. `top_driver` (the column) feeds the segmenting step.
    """
    rows = []
    values = contrib.to_numpy()
    cols = list(contrib.columns)
    raw = X[cols].to_numpy(dtype=object)  # row access via numpy: ~100× faster than .iloc
    top = np.argsort(-values, axis=1)[:, :k]
    for i in range(len(contrib)):
        positive = [j for j in top[i] if values[i, j] > 0]
        reasons = [reason_text(cols[j], raw[i, j], typical.get(cols[j])) for j in positive]
        rows.append({
            ID: ids.iloc[i],
            "churn_score": round(float(scores[i]), 4),
            "top_driver": cols[positive[0]] if positive else None,
            **{f"reason_{n + 1}": reasons[n] if n < len(reasons) else "" for n in range(k)},
        })
    return pd.DataFrame(rows)
