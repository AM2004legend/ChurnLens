# Progress

| Phase | Status | Notes |
|---|---|---|
| 0. Setup | ✅ done | venv (Python 3.12.6), pinned requirements, ruff + pytest config, folders, .gitignore, docs |
| 1. Dataset registry | ✅ done | 4 YAML configs; `io.load_config / read_raw / apply_mapping`; clear errors with close-match hints; xlsx parsed once and cached as pickle; all 4 datasets load |
| 2. Schema + cleaning | ✅ done | `schema.infer_types / missing_report / clean_labelled / clean_transactions`; reports in `reports/<name>/cleaning.json`. bank 10,000 rows (no missing); ecommerce 5,630 (4–6% missing in 7 numeric cols, 4 duplicate category spellings merged); subscription 243,787 (no missing); retail 1,067,371 → 779,424 rows, 5,878 customers |
| 3. Leakage guard + EDA | ✅ done | Stratified 80/20 split first; leakage + EDA on train only. Reports: `reports/<name>/leakage.md`, `eda.md`, figures in `reports/figures/`. Max single-feature AUC 0.80 — no leaks beyond bank RowNumber |
| 4. Transactions mode | ⏳ next | |
| 5. Modelling | ⏳ next | |
| 6. Cost-based threshold | | |
| 7. Explain + segments | | |
| 8. Cross-industry run | | |
| 9. Streamlit app | | |
| 10. Ship | | |

## Known issues
- Retail: churn rate shifts between train (64%) and test (49%) cutoffs — seasonality (test label window covers Sep–Dec Christmas trade). Probabilities will be miscalibrated on test; ranking metrics (ROC/PR-AUC) are less affected.
- Subscription dataset looks synthetic (categories almost perfectly uniform) — results show the method, not real behaviour.

## Later (ideas, not in scope yet)
- Survival analysis (lifelines) — PRD stretch, only after phase 10.
- Calibration plots — PRD stretch.
- Net retail returns against their original purchase line instead of only dropping the return row.
