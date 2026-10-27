import numpy as np
import pandas as pd
import pytest

from churnlens import explain, models, segments


@pytest.fixture
def fitted():
    rng = np.random.default_rng(3)
    n = 400
    x = pd.DataFrame({
        "inactive_days": rng.integers(0, 100, n).astype(float),
        "noise": rng.normal(size=n),
        "plan": rng.choice(["basic", "gold"], n).astype(object),
    })
    logit = 0.06 * (x["inactive_days"] - 50) + (x["plan"] == "basic") * 1.0
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return x, y


@pytest.mark.parametrize("name", ["logreg", "lightgbm"])
def test_shap_aggregates_to_business_columns(fitted, name: str) -> None:
    X, y = fitted
    num, cat = ["inactive_days", "noise"], ["plan"]
    model = models.fit(name, X, y, num, cat)
    values, names = explain.shap_matrix(model, X.head(50), X)
    assert values.shape == (50, len(names))
    contrib = explain.to_columns(values, names, num, cat)
    assert set(contrib.columns) == {"inactive_days", "noise", "plan"}
    # The planted driver must dominate the noise column.
    imp = explain.global_importance(contrib)
    assert imp.index[0] == "inactive_days"
    assert imp["noise"] < imp["inactive_days"]


def test_to_columns_sums_onehot_pieces() -> None:
    values = np.array([[0.1, 0.2, -0.05, 0.4]])
    names = ["num__age", "num__missingindicator_age", "cat__plan_a", "cat__plan_b"]
    out = explain.to_columns(values, names, ["age"], ["plan"])
    assert out.loc[0, "age"] == pytest.approx(0.3)
    assert out.loc[0, "plan"] == pytest.approx(0.35)


def test_explain_customers_uses_only_risk_raising_reasons() -> None:
    contrib = pd.DataFrame({"age": [0.5, -0.2], "plan": [0.2, -0.1], "calls": [-0.3, -0.4]})
    X = pd.DataFrame({"age": [58.0, 30.0], "plan": ["basic", "gold"], "calls": [1.0, 2.0]})
    out = explain.explain_customers(contrib, X, pd.Series(["c1", "c2"]), np.array([0.8, 0.1]),
                                    typical={"age": 37.0, "calls": 2.0})
    assert out.loc[0, "reason_1"] == "age is 58 (typical customer: 37)"
    assert out.loc[0, "reason_2"] == "plan is 'basic'"
    assert out.loc[0, "reason_3"] == ""  # calls lowered risk, so it is not a reason
    assert out.loc[0, "top_driver"] == "age"
    assert pd.isna(out.loc[1, "top_driver"])
    assert segments.rule_for(out.loc[1, "top_driver"], {})[0] == "No clear driver"


def test_reason_text_missing_value() -> None:
    assert explain.reason_text("Tenure", np.nan, 10.0) == "Tenure is not recorded"


def test_segments_assignment_and_summary() -> None:
    explained = pd.DataFrame({
        "customer_id": ["a", "b", "c", "d"],
        "churn_score": [0.9, 0.7, 0.6, 0.2],
        "top_driver": ["IsActiveMember", "Age", "IsActiveMember", "Age"],
    })
    rules = {"IsActiveMember": {"name": "Inactive members", "action": "Re-engage"}}
    seg = segments.assign_segments(explained, threshold=0.5, rules=rules)
    assert list(seg["customer_id"]) == ["a", "b", "c"]  # d is below threshold
    assert seg.set_index("customer_id").loc["a", "segment"] == "Inactive members"
    assert seg.set_index("customer_id").loc["b", "segment"] == "Driven by Age"
    assert seg.set_index("customer_id").loc["b", "suggested_action"] == segments.DEFAULT_ACTION
    summary = segments.segment_summary(seg)
    assert summary.iloc[0]["segment"] == "Inactive members" and summary.iloc[0]["customers"] == 2
    assert list(segments.targets_csv(seg).columns) == ["customer_id", "segment",
                                                       "suggested_action"]
