"""Train/test splitting and the preprocessing part of the model Pipeline.

Everything that *learns* from data (imputation values, category lists, scaling means) lives in
the ColumnTransformer, so `Pipeline.fit(X_train)` learns it from training rows only and the test
set is transformed with training statistics — the standard guard against preprocessing leakage.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnlens import config
from churnlens.io import TARGET

TEST_SIZE = 0.2
GROUP = "split_group"  # helper column, never a model feature


def group_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """One key per 'same customer' profile, e.g. identical cashback + demographics."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"split.group_by columns not in data: {missing}")
    return df[columns].astype("string").fillna("<NA>").agg("|".join, axis=1)


def split_labelled(
    df: pd.DataFrame, test_size: float = TEST_SIZE, groups: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split for one-row-per-customer data.

    Stratifying keeps the churn rate the same in train and test, which matters when churners
    are a minority (~15–20%). With `groups`, all rows sharing a group key go to the same side,
    so duplicated customers cannot be memorised in train and "recognised" in test.
    """
    if groups is None:
        train, test = train_test_split(
            df, test_size=test_size, stratify=df[TARGET], random_state=config.RANDOM_STATE
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)
    n_splits = round(1 / test_size)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df[TARGET], groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def build_preprocessor(
    numeric: list[str], categorical: list[str], scale: bool
) -> ColumnTransformer:
    """Impute + encode (+ scale for linear models).

    - numeric: median imputation (robust to skew, e.g. balances with many zeros), plus a
      missing-indicator column because "value not recorded" can itself predict churn.
    - categorical: missing becomes its own category, then one-hot; unseen test categories are
      ignored instead of crashing.
    - scaling only for logistic regression; trees are scale-invariant.
    """
    num_steps: list = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), numeric), ("cat", Pipeline(cat_steps), categorical)],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def source_column(feature_name: str, numeric: list[str], categorical: list[str]) -> str:
    """Map a transformed feature name back to the original column.

    e.g. 'cat__Geography_Germany' → 'Geography', 'num__missingindicator_Tenure' → 'Tenure'.
    Needed so SHAP explanations talk about business columns, not one-hot fragments.
    """
    prefix, _, rest = feature_name.partition("__")
    if prefix == "num":
        rest = rest.removeprefix("missingindicator_")
        return rest if rest in numeric else rest
    # Longest match wins so 'Plan' vs 'PlanType' can't be confused.
    matches = [c for c in categorical if rest == c or rest.startswith(f"{c}_")]
    return max(matches, key=len) if matches else rest
