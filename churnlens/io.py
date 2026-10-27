"""Load dataset configs and raw files, and map business-specific column names to standard ones.

Every later step works on standard names (`customer_id`, `churn`, `date`, `amount`), so the
same pipeline code runs on any dataset; only the YAML changes.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from churnlens import config

MODES = ("labelled", "transactions")

# Standard column names used by the rest of the pipeline.
ID = "customer_id"
TARGET = "churn"
REVENUE = "revenue"
DATE = "date"
AMOUNT = "amount"
INVOICE = "invoice"


class ConfigError(ValueError):
    """The YAML config itself is invalid (missing keys, wrong mode, bad values)."""


class MappingError(ValueError):
    """The config does not match the data (column missing, label value not found)."""


@dataclass
class Costs:
    """Business costs used by the threshold step. Defaults are assumptions, not real figures."""

    offer_cost: float = 500.0
    customer_value: float = 5000.0
    offer_success_rate: float = 0.3


@dataclass
class DatasetConfig:
    name: str
    mode: str
    path: Path
    columns: dict[str, Any]
    drop: list[str] = field(default_factory=list)
    costs: Costs = field(default_factory=Costs)
    read_options: dict[str, Any] = field(default_factory=dict)
    category_map: dict[str, dict[str, str]] = field(default_factory=dict)
    segments: dict[str, dict[str, str]] = field(default_factory=dict)
    transactions: dict[str, Any] = field(default_factory=dict)

    @property
    def inactivity_days(self) -> int:
        return int(self.transactions.get("inactivity_days", 90))

    @property
    def lookback_days(self) -> int:
        return int(self.transactions.get("lookback_days", 365))


def parse_config(raw: dict[str, Any], source: str = "<dict>") -> DatasetConfig:
    """Validate a config dict and turn it into a DatasetConfig.

    Fails early with a message naming the file and key, so a typo in YAML never surfaces later
    as a confusing pandas KeyError deep inside training.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: config must be a mapping, got {type(raw).__name__}")
    for key in ("name", "mode", "path", "columns"):
        if key not in raw:
            raise ConfigError(f"{source}: missing required key '{key}'")
    mode = raw["mode"]
    if mode not in MODES:
        raise ConfigError(f"{source}: mode must be one of {MODES}, got '{mode}'")

    columns = raw["columns"] or {}
    required = (
        ["customer_id", "churn_label", "churn_positive_value"]
        if mode == "labelled"
        else ["customer_id", "date", "amount"]
    )
    for key in required:
        if key not in columns or columns[key] is None:
            raise ConfigError(f"{source}: columns.{key} is required for mode '{mode}'")

    costs_raw = raw.get("costs") or {}
    unknown = set(costs_raw) - {"offer_cost", "customer_value", "offer_success_rate"}
    if unknown:
        raise ConfigError(f"{source}: unknown cost keys {sorted(unknown)}")
    costs = Costs(**{k: float(v) for k, v in costs_raw.items()})
    if not 0 <= costs.offer_success_rate <= 1:
        raise ConfigError(f"{source}: costs.offer_success_rate must be between 0 and 1")
    if costs.offer_cost < 0 or costs.customer_value < 0:
        raise ConfigError(f"{source}: costs must be non-negative")

    path = Path(raw["path"])
    if not path.is_absolute():
        path = config.ROOT / path

    return DatasetConfig(
        name=str(raw["name"]),
        mode=mode,
        path=path,
        columns=dict(columns),
        drop=list(raw.get("drop") or []),
        costs=costs,
        read_options=dict(raw.get("read") or {}),
        category_map=dict(raw.get("category_map") or {}),
        segments=dict(raw.get("segments") or {}),
        transactions=dict(raw.get("transactions") or {}),
    )


def load_config(name_or_path: str | Path) -> DatasetConfig:
    """Load `configs/datasets/<name>.yaml` (or an explicit YAML path)."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = config.CONFIGS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in config.CONFIGS_DIR.glob("*.yaml"))
        raise ConfigError(f"No config '{name_or_path}'. Available: {available}")
    with path.open(encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f), source=str(path))


def available_datasets() -> list[str]:
    return sorted(p.stem for p in config.CONFIGS_DIR.glob("*.yaml"))


def read_raw(cfg: DatasetConfig, use_cache: bool = True) -> pd.DataFrame:
    """Read the raw file named in the config without modifying it.

    Large Excel files are slow to parse (the retail workbook takes ~2 minutes), so the parsed
    frame is cached as a pickle in `data/processed/`. The cache is rebuilt whenever the raw
    file is newer than it.
    """
    if not cfg.path.exists():
        raise MappingError(
            f"Raw file for '{cfg.name}' not found at {cfg.path}. See data/README.md for sources."
        )
    suffix = cfg.path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(cfg.path, **cfg.read_options)
    if suffix not in (".xlsx", ".xls"):
        raise ConfigError(f"Unsupported file type '{suffix}' for {cfg.path}")

    cache = config.DATA_PROCESSED / f"{cfg.name}_raw.pkl"
    if use_cache and cache.exists() and cache.stat().st_mtime > cfg.path.stat().st_mtime:
        return pd.read_pickle(cache)

    options = dict(cfg.read_options)
    sheet = options.pop("sheet_name", 0)
    if sheet == "all":
        sheets = pd.read_excel(cfg.path, sheet_name=None, **options)
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_excel(cfg.path, sheet_name=sheet, **options)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def _missing_column_error(df: pd.DataFrame, role: str, column: str, dataset: str) -> MappingError:
    # Case-insensitive so "ID" still suggests "Id".
    by_lower = {str(c).lower(): str(c) for c in df.columns}
    close_lower = difflib.get_close_matches(column.lower(), list(by_lower), n=3, cutoff=0.6)
    close = [by_lower[c] for c in close_lower]
    hint = f" Did you mean {close}?" if close else ""
    return MappingError(
        f"[{dataset}] column '{column}' (mapped as {role}) is not in the data.{hint} "
        f"Columns available: {list(df.columns)}"
    )


def _require(df: pd.DataFrame, role: str, column: str, dataset: str) -> None:
    if column not in df.columns:
        raise _missing_column_error(df, role, column, dataset)


def _is_positive(series: pd.Series, positive: Any) -> pd.Series:
    """Compare label values to the 'churned' value, tolerant of 1 vs 1.0 vs '1' vs ' Yes'."""
    if pd.api.types.is_numeric_dtype(series) and not isinstance(positive, str):
        return series == positive
    as_text = series.astype("string").str.strip().str.lower()
    return as_text == str(positive).strip().lower()


def apply_mapping(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Rename mapped columns to standard names and encode the churn label as 0/1.

    Labelled mode keeps all other columns as candidate features. Transactions mode keeps only
    id, date, amount (and invoice if mapped) because features are built from them later.
    """
    cols = cfg.columns
    if cfg.mode == "labelled":
        return _map_labelled(df, cfg, cols)
    return _map_transactions(df, cfg, cols)


def _map_labelled(df: pd.DataFrame, cfg: DatasetConfig, cols: dict[str, Any]) -> pd.DataFrame:
    id_col, label_col = cols["customer_id"], cols["churn_label"]
    _require(df, "customer_id", id_col, cfg.name)
    _require(df, "churn_label", label_col, cfg.name)
    revenue_col = cols.get("revenue")
    if revenue_col:
        _require(df, "revenue", revenue_col, cfg.name)

    positive = cols["churn_positive_value"]
    label = df[label_col]
    is_pos = _is_positive(label, positive).fillna(False).astype(bool)
    if not is_pos.any():
        seen = label.dropna().unique()[:10].tolist()
        raise MappingError(
            f"[{cfg.name}] churn_positive_value {positive!r} never appears in '{label_col}'. "
            f"Values seen: {seen}"
        )

    out = df.copy()
    out[TARGET] = is_pos.astype("Int64").mask(label.isna())
    rename = {id_col: ID}
    if revenue_col:
        rename[revenue_col] = REVENUE
    out = out.drop(columns=[label_col]).rename(columns=rename)
    front = [ID, TARGET] + ([REVENUE] if revenue_col else [])
    return out[front + [c for c in out.columns if c not in front]]


def _map_transactions(df: pd.DataFrame, cfg: DatasetConfig, cols: dict[str, Any]) -> pd.DataFrame:
    _require(df, "customer_id", cols["customer_id"], cfg.name)
    _require(df, "date", cols["date"], cfg.name)
    amount = cols["amount"]
    amount_cols = amount if isinstance(amount, list) else [amount]
    for c in amount_cols:
        _require(df, "amount", c, cfg.name)
    invoice_col = cols.get("invoice")
    if invoice_col:
        _require(df, "invoice", invoice_col, cfg.name)

    out = pd.DataFrame({ID: df[cols["customer_id"]]})
    out[DATE] = pd.to_datetime(df[cols["date"]], errors="coerce")
    values = df[amount_cols].apply(pd.to_numeric, errors="coerce")
    # A list of columns means "multiply them", e.g. Quantity × Price = line amount.
    out[AMOUNT] = values.prod(axis=1, min_count=len(amount_cols))
    if invoice_col:
        out[INVOICE] = df[invoice_col].astype("string")
    return out
