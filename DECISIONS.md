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

## 2026-10-27 — Cleaning is rule-based only; imputation happens in the model Pipeline
- **Why:** a median computed on the whole dataset includes test rows → leakage. Cleaning only fixes labels and drops unusable rows (no id/label, duplicates).
- **Alternatives:** impute during cleaning (simpler, leaks).

## 2026-10-27 — Ecommerce: merge duplicate category spellings via `category_map` in YAML
- `Phone`→`Mobile Phone` (login device), `CC`→`Credit Card`, `COD`→`Cash on Delivery`, `Mobile`→`Mobile Phone` (order category).
- **Why:** otherwise one real category is split across two one-hot columns, weakening the signal and confusing explanations.

## 2026-10-27 — Retail: drop returns/cancellations, keep original purchase
- Rows with amount ≤ 0 or invoice starting with `C` are removed (18,460 rows).
- **Why:** a return is not a purchase; counting it would make customers look more active.
- **Trade-off:** the original purchase that was later returned still counts. Netting returns against purchases is more exact but needs line-level matching — listed as Later.

## 2026-10-27 — Retail: deduplicate on the full raw row, not the mapped columns
- **Why (bug found on real data):** after mapping to id/date/amount/invoice, two different products on the same invoice with the same line amount looked identical — 331,403 rows were wrongly dropped. Deduplicating on all raw columns drops only 26,480 true duplicates (mostly the overlapping Dec-2010 week in both sheets). Regression test added.
