import pandas as pd

from churnlens import io, schema


def _cfg(**extra) -> io.DatasetConfig:
    raw = {
        "name": "toy",
        "mode": "labelled",
        "path": "x.csv",
        "columns": {"customer_id": "Id", "churn_label": "Left", "churn_positive_value": 1},
    }
    raw.update(extra)
    return io.parse_config(raw)


def test_infer_types() -> None:
    df = pd.DataFrame(
        {
            "num": [1, 2, 3],
            "num_text": ["1", "2.5", "3"],
            "cat": ["a", "b", "a"],
            "date_text": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "real_date": pd.to_datetime(["2024-01-01"] * 3),
            "flag": [True, False, True],
        }
    )
    assert schema.infer_types(df, exclude=()) == {
        "num": "numeric",
        "num_text": "numeric",
        "cat": "categorical",
        "date_text": "date",
        "real_date": "date",
        "flag": "numeric",
    }


def test_missing_report_counts_blank_strings() -> None:
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": ["x", " ", None, "y"]})
    rep = schema.missing_report(df).set_index("column")
    assert rep.loc["b", "n_missing"] == 2
    assert rep.loc["a", "pct_missing"] == 25.0
    assert rep.index[0] == "b"  # worst first


def test_clean_labelled_maps_categories_and_drops_bad_rows() -> None:
    raw = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 3, 4, 5],
            "churn": pd.array([1, 0, 1, 1, None, 0], dtype="Int64"),
            "Device": [" Phone", "Mobile Phone", "", "", "Computer", "Phone"],
            "Age": [30, None, 40, 40, 50, 60],
        }
    )
    cfg = _cfg(category_map={"Device": {"Phone": "Mobile Phone"}})
    out, log = schema.clean_labelled(raw, cfg)
    assert log["dropped_missing_id_or_label"] == 1
    assert log["dropped_duplicate_rows"] == 1
    assert out["Device"].tolist() == ["Mobile Phone", "Mobile Phone", None, "Mobile Phone"]
    # Missing feature values are NOT imputed here — the Pipeline does that on train data only.
    assert out["Age"].isna().sum() == 1
    assert out["churn"].dtype == int
    assert out["customer_id"].tolist() == ["1", "2", "3", "5"]


def test_clean_labelled_dedupes_conflicting_ids() -> None:
    raw = pd.DataFrame({"customer_id": [7, 7], "churn": pd.array([1, 0], dtype="Int64"),
                        "x": [1, 2]})
    out, log = schema.clean_labelled(raw, _cfg())
    assert log["dropped_duplicate_ids"] == 1 and len(out) == 1


def test_clean_transactions_drops_returns_duplicates_and_unknown_customers() -> None:
    tx = pd.DataFrame(
        {
            "customer_id": [101.0, 101.0, 101.0, None, 102.0, 102.0],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-05", "2024-01-02", "2024-01-03",
                 "2024-01-04"]
            ),
            "amount": [10.0, 10.0, -10.0, 5.0, 7.0, 3.0],
            "invoice": ["A1", "A1", "C2", "A3", "A4", "C5"],
        }
    )
    out, log = schema.clean_transactions(tx)
    assert log["dropped_missing_customer"] == 1
    assert log["dropped_duplicate_rows"] == 1
    assert log["dropped_returns_or_zero_amount"] == 2  # negative amount + 'C' invoice
    assert out["customer_id"].tolist() == ["101", "102"]
    assert log["customers"] == 2


def test_clean_transactions_keeps_different_products_with_same_amount() -> None:
    # Two different products on one invoice, same time and amount: NOT duplicates.
    tx = pd.DataFrame(
        {
            "customer_id": [1.0, 1.0, 1.0],
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "amount": [5.0, 5.0, 5.0],
            "invoice": ["A1", "A1", "A1"],
            "raw_StockCode": ["X", "Y", "Y"],
        }
    )
    out, log = schema.clean_transactions(tx)
    assert log["dropped_duplicate_rows"] == 1
    assert len(out) == 2
    assert "raw_StockCode" not in out.columns
