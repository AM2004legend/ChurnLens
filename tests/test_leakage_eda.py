import numpy as np
import pandas as pd
import pytest

from churnlens import eda, features, leakage


@pytest.fixture
def leaky_df() -> pd.DataFrame:
    """200 customers with one planted leak of each kind."""
    rng = np.random.default_rng(0)
    n = 200
    churn = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "customer_id": [str(i) for i in range(n)],
            "churn": churn,
            "RowNumber": np.arange(n),  # ID-like
            "Salary": rng.normal(50_000, 10_000, n),  # continuous, unique, NOT an ID
            "DaysSinceCancel": churn * 30 + rng.normal(0, 1, n),  # near-perfect + bad name
            "Age": rng.integers(18, 80, n),
            "Surname": [f"name{i}" for i in range(n)],  # unique text: ID-like, but OOF AUC ~0.5
            "Plan": rng.choice(["a", "b"], n),
            "SignupDate": pd.date_range("2020-01-01", periods=n),
        }
    )


TYPES = {
    "RowNumber": "numeric",
    "Salary": "numeric",
    "DaysSinceCancel": "numeric",
    "Age": "numeric",
    "Surname": "categorical",
    "Plan": "categorical",
    "SignupDate": "date",
}


def _checks(report: leakage.LeakageReport) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for f in report.flags:
        out.setdefault(f.column, set()).add(f.check)
    return out


def test_leakage_flags_each_planted_problem(leaky_df: pd.DataFrame) -> None:
    checks = _checks(leakage.check_leakage(leaky_df, TYPES))
    assert checks["RowNumber"] == {"id_like"}
    assert checks["Surname"] == {"id_like"}
    assert checks["DaysSinceCancel"] == {"high_single_auc", "suspicious_name"}
    assert checks["SignupDate"] == {"date_column"}
    assert "Salary" not in checks  # unique floats are not IDs
    assert "Age" not in checks and "Plan" not in checks


def test_oof_encoding_does_not_reward_unique_text(leaky_df: pd.DataFrame) -> None:
    aucs = leakage.single_feature_aucs(leaky_df, TYPES)
    # In-sample target encoding would give Surname AUC = 1.0; out-of-fold gives ~chance.
    assert aucs["Surname"] < 0.65
    assert aucs["DaysSinceCancel"] > 0.95


def test_markdown_marks_undropped_flags(leaky_df: pd.DataFrame) -> None:
    report = leakage.check_leakage(leaky_df, TYPES)
    md = leakage.to_markdown(report, "toy", dropped=["RowNumber"])
    assert "| RowNumber | id_like | 1.0 | yes |" in md
    assert "| Surname | id_like | 1.0 | **no** |" in md


@pytest.fixture
def eda_df() -> pd.DataFrame:
    # Known answers: plan "b" churns 80% vs "a" 20%; churners are older.
    plan = ["a"] * 50 + ["b"] * 50
    churn = [1] * 10 + [0] * 40 + [1] * 40 + [0] * 10
    age = [60] * 10 + [30] * 40 + [60] * 39 + [61] + [30] * 10
    return pd.DataFrame({"churn": churn, "Plan": plan, "Age": age,
                         "Complaint": churn[:95] + [0] * 5})


def test_group_differences_known_rates(eda_df: pd.DataFrame) -> None:
    g = eda.group_differences(eda_df, ["Plan"], min_share=0.0)
    rates = dict(zip(g["value"], g["churn_rate"], strict=True))
    assert rates == {"a": 0.2, "b": 0.8}


def test_numeric_differences_direction(eda_df: pd.DataFrame) -> None:
    num = eda.numeric_differences(eda_df, ["Age", "Complaint"])
    assert num["column"].tolist() == ["Age"]  # binary column is analysed as groups instead
    assert num.loc[0, "median_churned"] == 60 and num.loc[0, "std_diff"] > 0


def test_findings_are_five_and_start_with_churn_rate(eda_df: pd.DataFrame) -> None:
    res = eda.run_eda(eda_df, {"Plan": "categorical", "Age": "numeric",
                               "Complaint": "numeric"}, "toy", save_figures=False)
    assert len(res.findings) == 4  # rate + Plan + Age + Complaint: only 3 columns exist
    assert res.findings[0].startswith("Overall, 50.0%")
    assert any("Plan = " in f for f in res.findings)


def test_stratified_split_keeps_churn_rate() -> None:
    df = pd.DataFrame({"churn": [1] * 20 + [0] * 80, "x": range(100)})
    train, test = features.split_labelled(df)
    assert len(test) == 20
    assert train["churn"].mean() == test["churn"].mean() == 0.2


def test_source_column_maps_back_onehot_and_indicator() -> None:
    num, cat = ["Tenure"], ["Plan", "PlanType"]
    assert features.source_column("cat__PlanType_gold", num, cat) == "PlanType"
    assert features.source_column("cat__Plan_gold", num, cat) == "Plan"
    assert features.source_column("num__missingindicator_Tenure", num, cat) == "Tenure"
