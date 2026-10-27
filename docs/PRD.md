# ChurnLens — Product Requirements

## Overview
A churn-analytics pipeline and app that works for any customer business. It predicts who is likely
to leave, explains why, and recommends whom to target using the business's own costs.

**Problem:** keeping a customer is usually far cheaper than winning a new one, but businesses don't
know who is about to leave or why, and most have only raw purchase history, not a "churned" label.

**Audience:** recruiters and interviewers for data science / analytics roles (IBM, Deloitte,
Accenture and similar). It shows the full workflow: framing, data quality, leakage, modelling,
explainability and a business decision.

**Goals**
1. One pipeline, many industries: proven on 4 public datasets.
2. Two input modes: labelled customer table, and raw transactions (label + RFM built automatically).
3. Correctness first: leakage guard, proper splits, honest metrics.
4. A business answer, not just a score: cost-based threshold, expected savings, segments with actions.

## Input modes
### A. Labelled customer table
One row per customer. User maps: `customer_id`, `churn_label` (and which value means "churned"),
optional `revenue` (value per customer). All other columns are candidate features.

### B. Raw transactions
One row per transaction. User maps: `customer_id`, `date`, `amount`. ChurnLens then:
1. Picks a **cutoff date** c (default: last date − 2 × window).
2. Uses only transactions on or before c to build features.
3. Labels a customer **churned** if they made **no transaction in (c, c + N days]** (N = inactivity
   window, default 90, configurable).
4. Includes only customers active in the 365 days before c (configurable).
5. Builds **RFM features**: recency (days since last purchase before c), frequency (purchases in
   lookback), monetary (total and average spend), plus tenure (days since first purchase), purchase
   trend (last 90 days vs previous 90).
6. **Time-based split:** train on cutoff c₁, test on a later cutoff c₂ so the model is judged on the
   future.

## Datasets (user downloads; record source + licence in data/README.md)
| Key | Industry | Dataset | Mode |
|---|---|---|---|
| bank | Banking | Bank Customer Churn — Kaggle `shrutimechlearn/churn-modelling` | A |
| ecommerce | E-commerce | E-commerce Customer Churn Analysis and Prediction — Kaggle `ankitverma2010/ecommerce-customer-churn-analysis-and-prediction` | A |
| subscription | Subscription / streaming | Predictive Analytics for Customer Churn — Kaggle `safrin03/predictive-analytics-for-customer-churn-dataset` | A |
| retail | Online retail | Online Retail II — UCI Machine Learning Repository | B |

If any dataset is unavailable, pick a replacement with me; don't substitute silently.

## Dataset config (one YAML per dataset)
```yaml
name: bank
mode: labelled            # labelled | transactions
path: data/raw/bank.csv
columns:
  customer_id: CustomerId
  churn_label: Exited
  churn_positive_value: 1
  revenue: null           # or a column name
drop: [RowNumber, Surname]  # decided after the leakage/EDA review
costs:
  offer_cost: 500         # cost of one retention offer (₹, assumed)
  customer_value: 5000    # value of keeping one customer (₹, assumed or from revenue)
  offer_success_rate: 0.3 # share of targeted churners the offer actually saves (assumed)
# transactions-mode only:
# date: InvoiceDate
# amount: [Quantity, Price]   # amount = Quantity × Price
# inactivity_days: 90
```
Assumed costs must be labelled as assumptions everywhere they appear.

## Features and acceptance criteria
| # | Feature | Acceptance criteria |
|---|---|---|
| 1 | Load + map | Any CSV loads via its YAML; wrong/missing column names give a clear error |
| 2 | Schema + cleaning | Numeric/categorical/date detection; missing-value report; cleaned file in data/processed |
| 3 | Leakage guard | Flags: ID-like columns (unique ratio > 0.95), single-feature ROC-AUC > 0.95, column names containing churn/cancel/exit/close/terminate, date columns that may be after the outcome. Report shown; user decides drops (recorded in YAML + DECISIONS.md) |
| 4 | EDA | Churn rate; top numeric and categorical differences between churners and stayers; 5 plain-English findings per dataset; figures saved |
| 5 | Transactions mode | Label from inactivity window; RFM + tenure + trend features; time-based split; unit tests on a hand-made 10-customer fixture with known labels |
| 6 | Modelling | Logistic regression (scaled, class_weight="balanced") vs LightGBM (class weights); 5-fold stratified CV on train; final test metrics: churn rate, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix; saved JSON |
| 7 | Cost-based threshold | For thresholds 0.05–0.95: expected savings = Σ over targeted customers of (is_churner × success_rate × customer_value − offer_cost). Choose max. Compare with "target everyone", "target no one", and threshold 0.5. Plot saved |
| 8 | Explainability | SHAP global importance plot; per-customer top 3 reasons in plain words |
| 9 | Segments + actions | At-risk customers grouped by their top SHAP driver into named segments (e.g. "Inactive recently", "Few products", "High complaints"), each with a suggested action from a rules table in config; downloadable CSV of targets (IDs only) |
| 10 | Cross-industry run | `python -m churnlens.run --all` runs all 4; `reports/results.md` table: dataset, rows, churn rate, best model, ROC-AUC, PR-AUC, recall at chosen threshold, expected savings (assumed costs) |
| 11 | Streamlit app | Pick demo dataset or upload CSV; map columns; leakage warnings; EDA; model comparison; threshold slider showing savings live; per-customer explanation; segment download |
| 12 | Ship | CI green; deployed link; README with results table, screenshots, limitations |

## Non-goals
Deep learning, AutoML, real-time scoring APIs, LLM-written insights, user accounts, storing uploaded
data after the session.

## Testing
- Unit tests with tiny synthetic fixtures for: mapping errors, typing, leakage flags, RFM and
  labelling (known answers), time split ordering, threshold maths (hand-calculated example), segment
  assignment.
- Smoke test: the full pipeline runs on a 200-row synthetic dataset in CI.
- No test depends on the Kaggle/UCI files (they're not in the repo).

## Honest limitations (README must state)
- Costs are assumptions; the method matters more than the rupee figures.
- Public datasets are partly synthetic or old; results show the method, not real-world performance.
- Inactivity-based churn depends on the chosen window; results change with it (show 60/90/120 days).
- Explanations show what the model uses, not proven causes.

## Interview story
"The same pipeline runs on banking, e-commerce, subscription and raw retail data. For retail I had to
define churn myself from inactivity, build RFM features and use a time-based split to avoid leakage.
I chose the threshold by expected savings instead of 0.5, which changed who gets targeted."

## Stretch (only after phase 10)
Survival analysis (time-to-churn) with lifelines on datasets with tenure; calibration plots.
