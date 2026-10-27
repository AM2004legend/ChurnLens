"""Smoke test: the whole pipeline on small synthetic data (no Kaggle/UCI files needed).

Uses the stage functions directly with figures off, so nothing is written to reports/.
"""

import pandas as pd

from churnlens import run, synthetic


def _end_to_end(cfg, raw: pd.DataFrame) -> dict:
    prep = run.prepare(cfg, raw=raw, save_figures=False)
    trained = run.train_and_evaluate(prep)
    decision = run.decide_threshold(prep, trained, save_figure=False)
    explained = run.explain_and_segment(prep, trained, decision, save_figure=False)
    return {"prep": prep, "trained": trained, "decision": decision, "explained": explained}


def test_labelled_pipeline_end_to_end() -> None:
    out = _end_to_end(synthetic.labelled_config(), synthetic.labelled(n=200))
    prep, trained, decision, explained = out.values()
    assert len(prep.train) == 160 and len(prep.test) == 40
    assert trained.best in ("logreg", "lightgbm")
    assert 0.05 <= decision.threshold <= 0.95
    m = decision.test_metrics_at_threshold
    assert {"precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"} <= set(m)
    assert set(explained.importance.index) == {"Age", "Plan", "TenureMonths", "SupportCalls",
                                               "MonthlyCharge"}
    assert explained.contributions.shape == (40, 5)
    assert len(explained.customers) == 40
    assert (explained.segmented["churn_score"] >= decision.threshold).all()


def test_transactions_pipeline_end_to_end() -> None:
    out = _end_to_end(synthetic.transactions_config(), synthetic.transactions(n_customers=200))
    prep = out["prep"]
    assert pd.Timestamp(prep.split_info["train_cutoff"]) < pd.Timestamp(
        prep.split_info["test_cutoff"])
    assert 0 < prep.split_info["train_churn_rate"] < 1
    assert "recency_days" in out["explained"].importance.index


def test_results_table_reads_saved_payload_fields() -> None:
    payload = {
        "dataset": "x", "mode": "labelled", "best_model": "logreg", "chosen_threshold": 0.4,
        "split": {"train_rows": 8, "test_rows": 2},
        "test_at_chosen_threshold": {"churn_rate": 0.5, "roc_auc": 0.8, "pr_auc": 0.7,
                                     "recall": 1.0, "precision": 0.5},
        "strategies_on_test": {"chosen_threshold": {"savings": 1234.0},
                               "threshold_0_5": {"savings": 1000.0},
                               "target_everyone": {"savings": -50.0}},
    }
    row = run.results_table([payload]).iloc[0]
    assert row["test ROC-AUC"] == "0.800" and row["expected savings (₹, ASSUMED costs)"] == "1,234"
