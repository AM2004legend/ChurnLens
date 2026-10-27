# ChurnLens

A reusable churn-analytics pipeline + Streamlit app: leakage guard, EDA, model comparison,
cost-based threshold, SHAP explanations and actionable segments. Works on labelled customer
tables and raw transaction logs.

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
ruff check .
```

See `docs/PRD.md` for the full spec and `PROGRESS.md` for status.
