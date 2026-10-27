# Decisions

Each entry: date — decision — why — alternatives considered.

## 2026-10-27 — Pin exact dependency versions
- **Why:** reproducible metrics; a changed LightGBM/sklearn default could silently change numbers in `reports/`.
- **Alternatives:** `>=` ranges (easier upgrades, less reproducible).

## 2026-10-27 — Add `openpyxl` and `matplotlib` to the stack
- **Why:** `openpyxl` is required by pandas to read the Online Retail II `.xlsx`; `matplotlib` is required by SHAP plots and saved EDA figures (PNG files for README).
- **Alternatives:** convert the xlsx to CSV by hand (extra manual step); plotly-only figures (SHAP's built-in plots are matplotlib).

## 2026-10-27 — Tests import the package via `tests/conftest.py` path insert
- **Why:** keeps setup to `pip install -r requirements-dev.txt`; no editable install needed on Windows.
- **Alternatives:** `pip install -e .` (cleaner imports, one more setup step).

## 2026-10-27 — Datasets downloaded by Claude from the PRD-listed sources
- **Why:** user asked Claude to proceed autonomously; Kaggle/UCI public download endpoints worked with the user's existing `~/.kaggle` setup. Same datasets as the PRD, no substitutes.
- **Alternatives:** manual browser download (identical files).

## 2026-10-27 — Subscription: use `train.csv` only
- **Why:** the Kaggle `test.csv` has no `Churn` column (competition hold-out), so it cannot be evaluated. We make our own train/test split from `train.csv`.

## 2026-10-27 — Transactions config: `date`/`amount`/`invoice` under `columns`, windows under `transactions`
- **Why:** column mappings stay in one place; `invoice` (optional) lets frequency count distinct orders rather than line items (retail has many lines per invoice).

## 2026-10-27 — Cache parsed Excel files as pickle in `data/processed/`
- **Why:** the retail workbook takes ~2 minutes to parse with openpyxl; caching makes reruns fast without touching `data/raw/`. Rebuilt automatically when the raw file is newer.
- **Alternatives:** parquet (needs pyarrow — a new dependency); CSV (loses datetime types).

## 2026-10-27 — Assumed costs differ per industry
- bank ₹500 offer / ₹5000 value; ecommerce ₹200 / ₹3000; subscription ₹150 / ₹2000; retail ₹300 / ₹4000; success rate 0.3 everywhere. **All are assumptions** — the method matters, not the figures.
