"""Hand-made 10-customer fixture with labels and features worked out by hand.

Cutoff c = 2024-06-30, inactivity window N = 90 days → label window (2024-06-30, 2024-09-28].
Lookback 365 days → a customer is included only with a purchase in (2023-07-01, 2024-06-30].
"""

import pandas as pd
import pytest

from churnlens import transactions as tr

CUTOFF = pd.Timestamp("2024-06-30")

ROWS = [
    # customer, date, amount, invoice
    ("A", "2024-06-01", 100, "a1"), ("A", "2024-07-15", 50, "a2"),        # comes back → 0
    ("B", "2024-01-10", 200, "b1"), ("B", "2024-05-01", 100, "b2"),        # silent → 1
    ("C", "2023-05-01", 80, "c1"),                                         # not in lookback → out
    ("D", "2024-08-01", 60, "d1"),                                         # new after c → out
    ("E", "2024-06-30 18:00", 40, "e1"), ("E", "2024-09-28 10:00", 40, "e2"),  # edges → 0
    ("F", "2024-06-15", 30, "f1"), ("F", "2024-09-29", 30, "f2"),          # back 1 day too late → 1
    ("G", "2023-07-02", 70, "g1"),                                         # in lookback → 1
    ("H", "2024-06-20", 10, "h1"), ("H", "2024-06-20", 20, "h2"), ("H", "2024-07-01", 5, "h3"),
    ("I", "2024-03-01", 10, "i1"), ("I", "2024-04-15", 10, "i2"), ("I", "2024-06-10", 10, "i3"),
    ("I", "2024-08-01", 10, "i4"),
    ("J", "2023-01-01", 500, "j1"), ("J", "2024-02-01", 50, "j2"), ("J", "2024-10-15", 50, "j3"),
]

EXPECTED_CHURN = {"A": 0, "B": 1, "E": 0, "F": 1, "G": 1, "H": 0, "I": 0, "J": 1}


@pytest.fixture
def tx() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=["customer_id", "date", "amount", "invoice"])
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["amount"] = df["amount"].astype(float)
    return df


@pytest.fixture
def snap(tx: pd.DataFrame) -> pd.DataFrame:
    return tr.build_snapshot(tx, CUTOFF, inactivity_days=90, lookback_days=365).set_index(
        "customer_id"
    )


def test_labels_match_hand_calculation(snap: pd.DataFrame) -> None:
    assert snap["churn"].to_dict() == EXPECTED_CHURN


def test_inactive_and_future_only_customers_excluded(snap: pd.DataFrame) -> None:
    assert "C" not in snap.index and "D" not in snap.index


def test_rfm_features_for_customer_b(snap: pd.DataFrame) -> None:
    b = snap.loc["B"]
    assert b["recency_days"] == 60          # May 1 → Jun 30
    assert b["frequency"] == 2
    assert b["monetary_total"] == 300
    assert b["monetary_avg"] == 150
    assert b["tenure_days"] == 172          # Jan 10 → Jun 30 (2024 is a leap year)
    assert b["purchases_last_90"] == 1      # (Apr 1, Jun 30]: May 1
    assert b["purchases_prev_90"] == 1      # (Jan 2, Apr 1]: Jan 10
    assert b["purchase_trend"] == 0


def test_features_ignore_purchases_after_cutoff(snap: pd.DataFrame) -> None:
    # A's July purchase must not leak into features.
    assert snap.loc["A", "monetary_total"] == 100
    assert snap.loc["A", "frequency"] == 1


def test_cutoff_day_purchase_counts_as_history(snap: pd.DataFrame) -> None:
    # E bought at 18:00 on the cutoff day: that is "on or before c".
    assert snap.loc["E", "recency_days"] == 0


def test_lookback_limits_frequency_but_not_tenure(snap: pd.DataFrame) -> None:
    j = snap.loc["J"]
    assert j["frequency"] == 1 and j["monetary_total"] == 50
    assert j["tenure_days"] == 546          # 2023-01-01 → 2024-06-30


def test_two_invoices_same_day_are_two_purchases(snap: pd.DataFrame) -> None:
    h = snap.loc["H"]
    assert h["frequency"] == 2 and h["monetary_avg"] == 15


def test_trend_positive_for_accelerating_customer(snap: pd.DataFrame) -> None:
    assert snap.loc["I", ["purchases_last_90", "purchases_prev_90", "purchase_trend"]].tolist() \
        == [2, 1, 1]


def test_invoice_lines_are_grouped(tx: pd.DataFrame) -> None:
    # Two lines on one invoice = one purchase.
    extra = pd.DataFrame({"customer_id": ["B"], "date": [pd.Timestamp("2024-05-01")],
                          "amount": [25.0], "invoice": ["b2"]})
    snap = tr.build_snapshot(pd.concat([tx, extra]), CUTOFF, 90, 365).set_index("customer_id")
    assert snap.loc["B", "frequency"] == 2 and snap.loc["B", "monetary_total"] == 325


def test_without_invoice_each_row_is_a_purchase(tx: pd.DataFrame) -> None:
    snap = tr.build_snapshot(tx.drop(columns="invoice"), CUTOFF, 90, 365)
    assert snap.set_index("customer_id")["churn"].to_dict() == EXPECTED_CHURN


def test_unobserved_label_window_is_rejected(tx: pd.DataFrame) -> None:
    # Data ends 2024-10-15; a cutoff of 2024-08-01 needs data until 2024-10-30.
    with pytest.raises(ValueError, match="label window"):
        tr.build_snapshot(tx, pd.Timestamp("2024-08-01"), 90, 365)


def test_time_split_orders_cutoffs(tx: pd.DataFrame) -> None:
    train, test, info = tr.time_split(tx, inactivity_days=90, lookback_days=365)
    c1, c2 = pd.Timestamp(info["train_cutoff"]), pd.Timestamp(info["test_cutoff"])
    last = pd.Timestamp(info["last_date"])
    assert c1 < c2
    assert c1 + pd.Timedelta(days=90) <= c2     # train labels are known before test cutoff
    assert c2 + pd.Timedelta(days=90) <= last   # test labels fully observed
    assert info["train_customers"] == len(train) and info["test_customers"] == len(test)


def test_default_cutoffs() -> None:
    c1, c2 = tr.default_cutoffs(pd.Timestamp("2024-12-31 15:00"), 90)
    assert c2 == pd.Timestamp("2024-10-02") and c1 == pd.Timestamp("2024-07-04")
