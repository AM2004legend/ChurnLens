# ChurnLens — project guide for Claude Code

## What this project is
ChurnLens is a reusable churn-analytics pipeline + Streamlit app. Give it customer data from any
business (bank, e-commerce, subscription, telecom, retail) and it:
1. maps and cleans the data,
2. guards against data leakage,
3. explains who churns and why (EDA),
4. trains and compares models,
5. picks a decision threshold from the business's own costs,
6. explains each prediction (SHAP) and groups at-risk customers into actionable segments.

It accepts **labelled customer tables** and **raw transaction logs** (where it builds the churn
label and RFM features itself). It is proven on four public datasets from different industries.
Full spec: `docs/PRD.md` — read it before any task.

## Who I am
- 3rd-year BTech CSBS student building this to LEARN data science properly. I'll be asked to explain
  every step in interviews, so understanding beats speed.
- I'm on Windows — give PowerShell commands.

---

## Workflow — follow this exactly, every phase

### At the start of every session
1. Read `CLAUDE.md`, `docs/PRD.md`, `PROGRESS.md` and `DECISIONS.md`.
2. Run `git status` and `git log --oneline -5`.
3. Tell me in 3–5 lines: where we are, what's next, anything broken.

### For each phase (see "Phases" below)
1. **Plan:** list files to create/change, functions with one-line purposes, tests to write, and any
   decision I need to make. Keep it short.
2. **Wait** for my "go". Do not write code before that.
3. **Implement** in small steps. Pure functions in `churnlens/`, thin UI in `app/`.
4. **Test:** write/extend pytest tests for the phase; run `pytest` and `ruff check .`; show the
   output. Fix failures before moving on.
5. **Verify on real data:** run the phase end-to-end on at least one dataset and show the actual
   output (row counts, metrics, a sample of results). Never describe results you didn't produce.
6. **Explain:** in plain language — what was built, the 2–3 data-science concepts involved, and 2
   interview questions I should be able to answer. Mention anything surprising in the data.
7. **Record:** update `PROGRESS.md` (phase done, what works, known issues) and append any choice
   with trade-offs to `DECISIONS.md` (date, decision, why, alternatives).
8. **Commit** (see Git rules), then **stop** and wait for me.

### When something fails
- Show the exact error, explain the cause in plain words, then propose a fix. Don't silently retry
  different approaches.
- If stuck after 2 attempts, stop and ask me.

---

## Guardrails

### Scope
- Build only what's in the PRD for the current phase. Ideas go in a "Later" list at the end of
  `PROGRESS.md`, not into code.
- No deep learning, no AutoML libraries, no LLM calls in the pipeline. scikit-learn + LightGBM + SHAP.
- Don't add a new dependency without telling me why and getting a yes.

### Data
- Raw datasets live in `data/raw/` and are **gitignored**. Never commit data files, models over
  1 MB, or generated reports with customer rows.
- I download datasets myself (Kaggle needs login). Never scrape or use a dataset not listed in the
  PRD without asking. Note each dataset's source and licence in `data/README.md`.
- Never modify files in `data/raw/`. All cleaning writes to `data/processed/`.
- Only public or synthetic data. No real personal data.

### Machine-learning correctness (most important)
- **Split before anything is fitted.** Imputers, encoders, scalers live inside a scikit-learn
  `Pipeline` fitted on training data only.
- **Leakage guard runs before training** and its warnings are shown to me; I decide what to drop.
- **Transactions mode uses time-based splits** (train on an earlier cutoff, test on a later one).
  Never random-split time-dependent data.
- **The test set is touched once**, at the end of a phase, for the final numbers. Tuning uses
  cross-validation on training data only.
- Fixed random seeds everywhere (`RANDOM_STATE = 42` in `churnlens/config.py`).
- Always report: churn rate, precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix at the
  chosen threshold. Never report accuracy alone.
- **Never invent or round up metrics.** Every number in README/explanations comes from a saved file
  in `reports/` produced by code.
- If a result looks too good (e.g. AUC > 0.97), stop and look for leakage before celebrating.

### Code quality
- Python 3.12, type hints on public functions, docstrings explaining *why* not just *what*.
- `ruff` clean; `pytest` green before every commit.
- Config over hard-coding: each dataset has a YAML in `configs/datasets/`.
- Functions small enough to unit-test; no logic buried in Streamlit callbacks.

### Safety
- Never delete files outside this project. Ask before deleting anything not created this session.
- No secrets are needed for this project; if any appear, they go in `.env` (gitignored).

---

## Git rules
- Commit after each phase (and at sensible points within a phase), only when tests pass.
- Commit messages: Conventional Commits, e.g. `feat(leakage): flag ID-like and near-perfect columns`,
  `test(rfm): cover customers with a single purchase`, `docs: add cross-industry results table`.
- **Author every commit as me only. Do NOT add Claude or Anthropic as a co-author or contributor:**
  - no `Co-Authored-By:` trailer of any kind,
  - no "Generated with Claude Code" or similar line in commit messages or PR descriptions,
  - never change `git config user.name` / `user.email`, never pass `--author`.
- Never force-push, rewrite published history, or push without my explicit OK.
- Never commit `data/raw/`, `.env`, `.venv/`, large models or notebooks' output cells with data rows.

---

## Tech stack
Python 3.12 · pandas · numpy · scikit-learn · LightGBM · SHAP · PyYAML · Streamlit · plotly ·
joblib · pytest · ruff. Virtual env in `.venv`.

## Folder structure
```
churnlens/
  config.py          # seeds, paths, defaults
  io.py              # load CSV, apply column mapping
  schema.py          # column typing, cleaning
  leakage.py         # leakage guard
  eda.py             # EDA summary + plain-English findings
  transactions.py    # churn label from inactivity + RFM features + time split
  features.py        # sklearn ColumnTransformer / Pipeline
  models.py          # train, cross-validate, evaluate, save
  threshold.py       # cost-based threshold + expected value
  explain.py         # SHAP global + per-customer
  segments.py        # reason-based segments + suggested actions
  run.py             # CLI: python -m churnlens.run --dataset bank
app/streamlit_app.py
configs/datasets/    # bank.yaml, ecommerce.yaml, subscription.yaml, retail.yaml
data/raw/ data/processed/   (gitignored)
reports/             # metrics JSON, figures, results tables (small, no customer rows)
tests/               # unit tests + tiny synthetic fixtures
PROGRESS.md  DECISIONS.md  docs/PRD.md
```

## Phases (stop after each)
0. **Setup:** venv, requirements, ruff, pytest, folders, `.gitignore`, `.claude/settings.json`
   attribution check, `PROGRESS.md`, `DECISIONS.md`, `data/README.md`. First commit.
1. **Dataset registry:** YAML config per dataset (path, mode, column mapping, costs); loader that
   applies the mapping; I download the 4 datasets. Tests with tiny fixtures.
2. **Schema + cleaning:** automatic typing, missing values report, cleaning to `data/processed/`.
3. **Leakage guard + EDA:** leakage report; EDA with churn rate, top differences, 5 plain-English
   findings per dataset.
4. **Transactions mode:** inactivity-based label, RFM features, time-based train/test cutoffs.
5. **Modelling:** logistic regression baseline vs LightGBM, stratified CV, class weights, final test
   metrics saved to `reports/`.
6. **Cost-based threshold:** expected-value curve, chosen threshold, savings vs "target everyone"
   and "target no one".
7. **Explain + segments:** SHAP global + per-customer; reason-based segments with suggested actions.
8. **Cross-industry run:** all 4 datasets end to end via `python -m churnlens.run --all`; results
   table in `reports/results.md`.
9. **Streamlit app:** upload or pick a demo dataset, map columns, see leakage warnings, EDA, model
   comparison, threshold slider with savings, SHAP per customer, segment list download.
10. **Ship:** CI (ruff + pytest), deploy, README with results table and screenshots, 2-minute demo
    script.
