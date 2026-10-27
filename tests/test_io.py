from pathlib import Path

import pandas as pd
import pytest

from churnlens import io


def _labelled_raw(**overrides) -> dict:
    raw = {
        "name": "toy",
        "mode": "labelled",
        "path": "data/raw/toy.csv",
        "columns": {"customer_id": "Id", "churn_label": "Left", "churn_positive_value": "Yes"},
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "Left": ["Yes", "No", " yes", None],
            "Age": [30, 40, 50, 60],
        }
    )


def test_parse_config_defaults_costs() -> None:
    cfg = io.parse_config(_labelled_raw())
    assert cfg.costs.offer_cost == 500
    assert cfg.path.is_absolute()


def test_parse_config_rejects_bad_mode() -> None:
    with pytest.raises(io.ConfigError, match="mode must be one of"):
        io.parse_config(_labelled_raw(mode="labeled"))


def test_parse_config_requires_label_for_labelled_mode() -> None:
    raw = _labelled_raw(columns={"customer_id": "Id"})
    with pytest.raises(io.ConfigError, match="churn_label"):
        io.parse_config(raw)


def test_parse_config_rejects_success_rate_above_one() -> None:
    with pytest.raises(io.ConfigError, match="offer_success_rate"):
        io.parse_config(_labelled_raw(costs={"offer_success_rate": 1.5}))


def test_mapping_renames_and_encodes_label(toy_df: pd.DataFrame) -> None:
    cfg = io.parse_config(_labelled_raw())
    out = io.apply_mapping(toy_df, cfg)
    assert list(out.columns[:2]) == ["customer_id", "churn"]
    # " yes" counts as positive (case/space tolerant); missing label stays missing.
    assert out["churn"].tolist()[:3] == [1, 0, 1]
    assert pd.isna(out["churn"].iloc[3])
    assert "Left" not in out.columns
    assert "Age" in out.columns


def test_mapping_missing_column_suggests_close_match(toy_df: pd.DataFrame) -> None:
    raw = _labelled_raw(
        columns={"customer_id": "ID", "churn_label": "Left", "churn_positive_value": "Yes"}
    )
    with pytest.raises(io.MappingError, match=r"Did you mean \['Id'\]"):
        io.apply_mapping(toy_df, io.parse_config(raw))


def test_mapping_positive_value_not_found(toy_df: pd.DataFrame) -> None:
    raw = _labelled_raw(
        columns={"customer_id": "Id", "churn_label": "Left", "churn_positive_value": "Churned"}
    )
    with pytest.raises(io.MappingError, match="never appears"):
        io.apply_mapping(toy_df, io.parse_config(raw))


def test_numeric_positive_value_matches_float_column() -> None:
    df = pd.DataFrame({"Id": [1, 2], "Left": [1.0, 0.0]})
    raw = _labelled_raw(
        columns={"customer_id": "Id", "churn_label": "Left", "churn_positive_value": 1}
    )
    out = io.apply_mapping(df, io.parse_config(raw))
    assert out["churn"].tolist() == [1, 0]


def test_transactions_mapping_multiplies_amount_columns() -> None:
    df = pd.DataFrame(
        {
            "Cust": [1, 1],
            "When": ["2024-01-01", "2024-02-01"],
            "Qty": [2, 3],
            "Price": [5.0, 1.5],
            "Inv": ["A1", "A2"],
        }
    )
    raw = {
        "name": "tx",
        "mode": "transactions",
        "path": "x.csv",
        "columns": {"customer_id": "Cust", "date": "When", "amount": ["Qty", "Price"],
                    "invoice": "Inv"},
    }
    out = io.apply_mapping(df, io.parse_config(raw))
    assert out["amount"].tolist() == [10.0, 4.5]
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert list(out.columns) == ["customer_id", "date", "amount", "invoice"]


def test_transactions_mapping_keeps_raw_columns_for_dedup() -> None:
    df = pd.DataFrame({"Cust": [1], "When": ["2024-01-01"], "Amt": [5.0], "Sku": ["X"]})
    raw = {"name": "tx", "mode": "transactions", "path": "x.csv",
           "columns": {"customer_id": "Cust", "date": "When", "amount": "Amt"}}
    out = io.apply_mapping(df, io.parse_config(raw))
    assert "raw_Sku" in out.columns


def test_read_raw_csv_roundtrip(tmp_path: Path, toy_df: pd.DataFrame) -> None:
    path = tmp_path / "toy.csv"
    toy_df.to_csv(path, index=False)
    cfg = io.parse_config(_labelled_raw(path=str(path)))
    assert io.read_raw(cfg).shape == toy_df.shape


def test_read_raw_missing_file_is_clear(tmp_path: Path) -> None:
    cfg = io.parse_config(_labelled_raw(path=str(tmp_path / "nope.csv")))
    with pytest.raises(io.MappingError, match="not found"):
        io.read_raw(cfg)


@pytest.mark.parametrize("name", ["bank", "ecommerce", "subscription", "retail"])
def test_shipped_configs_are_valid(name: str) -> None:
    cfg = io.load_config(name)
    assert cfg.name == name
