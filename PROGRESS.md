# Progress

| Phase | Status | Notes |
|---|---|---|
| 0. Setup | ✅ done | venv (Python 3.12.6), pinned requirements, ruff + pytest config, folders, .gitignore, docs |
| 1. Dataset registry | ✅ done | 4 YAML configs; `io.load_config / read_raw / apply_mapping`; clear errors with close-match hints; xlsx parsed once and cached as pickle; all 4 datasets load |
| 2. Schema + cleaning | ✅ done | `schema.infer_types / missing_report / clean_labelled / clean_transactions`; reports in `reports/<name>/cleaning.json`. bank 10,000 rows (no missing); ecommerce 5,630 (4–6% missing in 7 numeric cols, 4 duplicate category spellings merged); subscription 243,787 (no missing); retail 1,067,371 → 779,424 rows, 5,878 customers |
| 3. Leakage guard + EDA | ✅ done | Stratified 80/20 split first; leakage + EDA on train only. Reports: `reports/<name>/leakage.md`, `eda.md`, figures in `reports/figures/`. Max single-feature AUC 0.80 — no leaks beyond bank RowNumber |
| 4. Transactions mode | ⏳ next | |
| 5. Modelling | ✅ done | `models.py` + `run.py` (`python -m churnlens.run --dataset X / --all`). LogReg vs LightGBM, 5-fold stratified (grouped where configured) CV on train, best by CV PR-AUC, test metrics @0.5 in `reports/<name>/metrics.json`. Best test ROC-AUC / PR-AUC: bank LightGBM 0.856/0.700, ecommerce LightGBM 0.943/0.830, retail LogReg 0.752/0.700, subscription LogReg 0.755/0.409 |
| 6. Cost-based threshold | ✅ done | `threshold.py`; threshold chosen on train OOF scores, savings reported on test vs everyone / no one / 0.5 (`reports/<name>/threshold.json`, figure). Test savings (ASSUMED costs): bank ₹173.5k @0.60, ecommerce ₹97.1k @0.19, retail ₹1.372M @0.18, subscription ₹980k @0.59 — all beat 0.5 and both naive strategies |
| 7. Explain + segments | ✅ done | `explain.py` (SHAP Tree/Linear, one-hot pieces summed back to business columns, top-3 risk-raising reasons in plain words) + `segments.py` (top driver → YAML rules → segment + action). Aggregates in `reports/<name>/explain.json`; per-customer reasons and target IDs written to gitignored CSVs. Top drivers: bank Age/NumOfProducts, ecommerce Tenure/Complain, retail recency/frequency, subscription AccountAge/viewing duration |
| 8. Cross-industry run | ✅ done | `python -m churnlens.run --all` runs all 4 (~2 min) and writes `reports/results.md` (results table, model comparison, confusion matrices, SHAP drivers, retail 60/90/120-day sensitivity). Smoke tests run the full pipeline on 200-row synthetic labelled + transactions data |
| 9. Streamlit app | ✅ done | `streamlit run app/streamlit_app.py`. Synthetic demos (always available), local real demos (if raw files exist), or CSV upload with column mapping. Tabs: Overview, Leakage (user picks drops), EDA, Models, Threshold (live slider + savings), Explain (per customer), Segments (CSV download). Headless `AppTest` covers load → train → all tabs; checked manually in Chrome on bank (₹173,500 @0.60, matches CLI) |
| 10. Ship | 🟡 mostly done | GitHub Actions CI (ruff + pytest, no data needed); README (results table copied by `scripts/build_readme.py` from `reports/results.md`, figures, screenshot, limitations); `docs/DEMO_SCRIPT.md`. **Remaining (needs Aritra):** create GitHub repo + push, confirm CI green, deploy on Streamlit Community Cloud, add the link to README |

## Known issues
- Deployed app will show only the synthetic demos (raw datasets are gitignored); real datasets work locally.
- Retail: churn rate shifts between train (64%) and test (49%) cutoffs — seasonality (test label window covers Sep–Dec Christmas trade). Probabilities will be miscalibrated on test; ranking metrics (ROC/PR-AUC) are less affected.
- Ecommerce: ~66% of rows are near-copies of other customers; handled with a grouped split. Its scores are still the highest of the four and should be read with that caveat.
- Retail: CV ROC-AUC 0.80 vs test 0.75 — the gap is the seasonal shift between cutoffs.
- Subscription dataset looks synthetic (categories almost perfectly uniform) — results show the method, not real behaviour.

## Later (ideas, not in scope yet)
- Survival analysis (lifelines) — PRD stretch, only after phase 10.
- Calibration plots — PRD stretch.
- Hyper-parameter tuning with CV on train (e.g. small LightGBM grid).
- Net retail returns against their original purchase line instead of only dropping the return row.
