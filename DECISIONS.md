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

## 2026-10-27 — Split before leakage guard and EDA
- **Why:** looking at test-set labels (even in EDA) influences feature decisions, a subtle leak. Leakage checks and EDA use the 80% training split only.

## 2026-10-27 — Leakage guard details
- ID-like check skips continuous floats (EstimatedSalary is 99.99% unique but is not an ID); whole-number floats are still checked.
- Categorical single-feature AUC uses **out-of-fold** target encoding (5 folds). In-sample encoding would give any unique text column (Surname) AUC ≈ 1.0 — a false alarm.

## 2026-10-27 — Drops after the leakage/EDA review
- **bank:** drop `RowNumber` (flagged id_like, 100% unique) and `Surname` (not flagged — 29% unique — but a personal name has no causal meaning and could act as a nationality/ethnicity proxy).
- **ecommerce:** none. `Complain` kept: plausible pre-churn signal, single AUC 0.65 (far from leak territory). Caveat: we cannot verify when it was recorded.
- **subscription:** none.

## 2026-10-27 — Transactions-mode definitions
- **Day level:** a purchase at 18:00 on the cutoff day counts as history ("on or before c").
- **Purchase = invoice** when an invoice column is mapped (lines summed), otherwise each row.
- **Cutoffs:** c2 = last − N (test labels fully observed), c1 = c2 − N = last − 2N (PRD default). Train labels end exactly at c2, so every test label lies in the future of all training data.
- **Guard:** building a snapshot whose label window runs past the end of the data raises an error — otherwise everyone would look churned simply because their future is unobserved.
- **Trend** = purchases in last 90 days − purchases in previous 90 (simple, interpretable, testable). Alternative: ratio (undefined for 0/0).

## 2026-10-27 — Same customer may appear in train and test snapshots (transactions mode)
- **Why:** realistic — customers are re-scored over time. The split is by time, not by customer, so test labels are always in the future. Alternative: also hold out customers (group split), which measures new-customer generalisation and halves the data.

## 2026-10-27 — Retail window sensitivity (60/90/120 days)
| N | train cutoff | test cutoff | train churn | test churn |
|---|---|---|---|---|
| 60 | 2011-08-11 | 2011-10-10 | 65.6% | 56.0% |
| 90 | 2011-06-12 | 2011-09-10 | 64.1% | 49.3% |
| 120 | 2011-04-13 | 2011-08-11 | 57.6% | 46.5% |
- The label depends on N, and train/test rates differ because the windows fall in different seasons. Default N = 90 kept.

## 2026-10-27 — Date parsing uses `format="mixed"`
- **Why:** with `errors="coerce"` alone, pandas infers one format from the first value and silently turns differently-formatted dates into NaT (found via the test fixture). Mixed parsing handles uploaded CSVs with inconsistent date strings.

## 2026-10-27 — Ecommerce near-duplicate customers → grouped split (leak found in phase 5)
- **Symptom:** LightGBM test ROC-AUC 0.9985 / PR-AUC 0.992 while logistic regression got 0.89 — the "too good to be true" rule fired.
- **Cause:** 557 rows (9.9%) repeat another row's entire feature profile under a new CustomerID; ~66% of rows share CashbackAmount (to the cent) + Gender + MaritalStatus + CityTier with another row, and 98.7% of those groups have one label. Likely oversampled copies. A random split puts copies on both sides; trees memorise them.
- **Evidence (CV on train only):** random CV LightGBM AUC 0.989 → grouped CV 0.958; logreg 0.896 → 0.893 (linear models can't memorise, so they barely move).
- **Fix:** new leakage check `duplicate_profiles` (flags > 1% duplicate feature rows) and YAML `split.group_by`; train/test split uses StratifiedGroupKFold and CV uses grouped folds. Result: test ROC-AUC 0.943, PR-AUC 0.830, consistent with CV 0.954.
- **Alternatives:** drop exact duplicates only (removes 557 rows but near-copies remain: CV AUC still 0.987); de-duplicate by the group key (throws away 1,470 rows of possibly-real customers).

## 2026-10-27 — Model settings
- LightGBM: 300 trees, lr 0.05, 31 leaves, min_child_samples 30, row/col subsampling 0.8, class_weight balanced. No hyper-parameter search — the PRD asks for a fair baseline comparison; tuning is a Later item (would use CV on train only).
- LogReg: StandardScaler + class_weight balanced, max_iter 2000.
- Missing numeric → median + missing-indicator columns; missing categorical → "Missing" category. Date columns are not used as raw features.
- **Selection by CV PR-AUC**, never by test. Test metrics for both models are reported for transparency.
- **Class weights shift probabilities upward** (they are not calibrated churn probabilities). That is fine for ranking and for the cost-based threshold, which is chosen on out-of-fold scores against real labels.

## 2026-10-27 — Findings worth noting
- LightGBM clearly beats LogReg on bank (PR-AUC 0.70 vs 0.47: non-linear effects like NumOfProducts = 3–4) and ecommerce. On retail RFM and subscription, LogReg is as good or slightly better — few, mostly monotonic features leave nothing for trees to add.

## 2026-10-27 — Threshold chosen on out-of-fold training scores
- **Why:** choosing it on the test set would make the reported savings optimistic (we'd be tuning on the exam). OOF scores come from models that never saw those rows, so they behave like unseen data.
- Grid 0.05–0.95 step 0.01; ties → higher threshold (same money, fewer offers).
- **Break-even reference:** with calibrated probabilities, target when p > c/(s·v) (bank: 500/(0.3×5000) = 0.33). Our chosen thresholds differ (bank 0.60) because class weighting inflates scores — one reason to pick empirically. Calibration plots are a stretch item.
- Customer value is a single assumed number per dataset; `expected_savings` accepts per-customer values for when a trustworthy revenue column exists.

## 2026-10-27 — Retail: "target everyone" is profitable
- With 49% churn on test and an offer costing ₹300 vs ₹1,200 expected recovery per saved churner, mass targeting pays (₹1.257M). The model still adds value: ₹1.372M while sending 551 fewer offers. Worth stating honestly — when churn is very common and offers are cheap, the model's job is pruning, not finding.

## 2026-10-27 — SHAP setup
- TreeExplainer for LightGBM, LinearExplainer (1,000-row training background) for LogReg; both in log-odds.
- SHAP is computed on the transformed features, then **summed back to the business column** (one-hot pieces + missing indicators). Valid because SHAP values are additive.
- Explained set = test customers (the ones a team would act on).
- **Reasons use only positive contributions** (what raises risk), compared with the training median ("typical customer"). Explanations describe the model, not causes.

## 2026-10-27 — Segments = top risk driver, named by YAML rules
- Customers at/above the chosen threshold are grouped by their single largest positive SHAP column. Names/actions live in each YAML so the business owns the wording; unmapped drivers get "Driven by X" + a generic action. Rule names were set after checking driver direction on real data (e.g. bank Age-driven customers are 49–61 vs typical 37).
- Suggested actions are **illustrative business rules, not tested interventions**.
- Driver = NaN (no positive contribution) → "No clear driver" (bug caught by a unit test: pandas turns None into NaN).
- Targets CSV holds only customer_id, segment, action (no attributes) and is gitignored, as is the per-customer reasons file.

## 2026-10-27 — Results table generated from saved payloads only
- `reports/results.md` is written by code from the same dicts saved as `reports/<name>/metrics.json` — no hand-typed numbers (CLAUDE.md rule).
- Window sensitivity re-runs modelling + threshold in memory for N = 60/90/120 (no extra files). Finding: AUCs stay 0.72–0.77 but churn rate, chosen threshold and savings move a lot (₹1.70M → ₹1.27M) — the label definition is a business decision, not a detail.

## 2026-10-27 — App design
- **Two-step flow** (Load & check → review leakage → Train) so the user decides drops before any model is fitted, matching the CLAUDE.md rule "leakage guard runs before training; I decide what to drop".
- **Synthetic demo datasets** (`churnlens/synthetic.py`): raw Kaggle/UCI files are gitignored, so a deployed app would otherwise have no data. Real datasets appear as demos only where the files exist locally.
- **Uploads stay in memory**: `prepare(raw=df, save_figures=False)`, no model or report files written (PRD non-goal: storing uploaded data).
- Threshold slider recomputes savings live on test scores; the *recommended* threshold is recomputed from train OOF scores when costs change, so it never uses test labels.
- `AppTest` drives the real script in CI — catches broken tabs without a browser.

## 2026-10-27 — README numbers are copied by script
- `scripts/build_readme.py` fills `docs/README_template.md` with the tables from `reports/results.md`, so README numbers cannot drift from the code's output. Re-run after `python -m churnlens.run --all`.

## 2026-10-27 — Deployment target: Streamlit Community Cloud
- Free, deploys straight from the GitHub repo using `requirements.txt`; main file `app/streamlit_app.py`. Needs the user's GitHub + Streamlit accounts, so Claude prepares everything but does not push or deploy.
- Alternatives: Hugging Face Spaces (also free; Docker or Streamlit SDK), Render (needs a start command and sleeps on free tier).

## 2026-10-27 — Container-aware thread count (deployed app was ~40× slow)
- **Symptom:** on Streamlit Cloud, training the 2,000-row synthetic demo had not finished after 2+ minutes; locally it takes ~3 s and the full test suite (incl. an app training run) takes ~16 s on GitHub's Linux runners.
- **Cause (hypothesis, consistent with both observations):** `n_jobs=-1` lets LightGBM/BLAS start one thread per core *reported* by the OS, but the container's cgroup CPU quota is much smaller, so threads oversubscribe the real CPUs.
- **Fix:** `config.n_jobs()` = min(visible cores, cgroup v2 `cpu.max` / v1 cfs quota), overridable via `CHURNLENS_N_JOBS`; used for LightGBM, and the app caps OMP/OpenBLAS/MKL threads to the same value before numpy loads. Results unchanged locally (bank metrics.json byte-identical).
