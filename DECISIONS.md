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
