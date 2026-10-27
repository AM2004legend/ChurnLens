import numpy as np
import pandas as pd
import pytest

from churnlens import models


@pytest.fixture
def toy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(size=n)
    plan = rng.choice(["a", "b", "c"], n)
    logit = 1.5 * x1 + (plan == "c") * 1.0 - 1.5
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    x1[:10] = np.nan  # missing values must be handled inside the pipeline
    X = pd.DataFrame({"x1": x1, "plan": plan.astype(object)})
    return X, y


def test_evaluate_known_confusion_matrix() -> None:
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
    m = models.evaluate(y, p, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["churn_rate"] == 0.4 and m["share_targeted"] == 0.4
    assert "accuracy" not in m


def test_evaluate_no_one_targeted_gives_zero_not_error() -> None:
    m = models.evaluate(np.array([1, 0]), np.array([0.2, 0.1]), threshold=0.5)
    assert m["precision"] == 0 and m["recall"] == 0


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_models_fit_with_missing_and_unseen_categories(toy, name: str) -> None:
    X, y = toy
    model = models.fit(name, X, y, ["x1"], ["plan"])
    new = pd.DataFrame({"x1": [np.nan, 0.5], "plan": ["zzz", "a"]})  # unseen category
    p = model.predict_proba(new)[:, 1]
    assert p.shape == (2,) and np.all((p >= 0) & (p <= 1))


def test_cross_validate_returns_oof_for_every_row(toy) -> None:
    X, y = toy
    table, oof = models.cross_validate(X, y, ["x1"], ["plan"])
    assert set(table["model"]) == set(models.MODEL_NAMES)
    for scores in oof.values():
        assert len(scores) == len(y) and np.all(scores > 0)
    assert table["roc_auc_mean"].min() > 0.6  # real signal was planted
    assert models.choose_best(table) in models.MODEL_NAMES


def test_cross_validate_is_reproducible(toy) -> None:
    X, y = toy
    a, _ = models.cross_validate(X, y, ["x1"], ["plan"], names=("lightgbm",))
    b, _ = models.cross_validate(X, y, ["x1"], ["plan"], names=("lightgbm",))
    pd.testing.assert_frame_equal(a, b)
