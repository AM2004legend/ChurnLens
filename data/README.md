# Data

Raw files go in `data/raw/` (gitignored, never modified). Cleaned outputs go in `data/processed/` (gitignored).

| Key | File | Source | Licence | Mode |
|---|---|---|---|---|
| bank | `data/raw/bank.csv` | Kaggle — [shrutimechlearn/churn-modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) | As listed on Kaggle page (check before redistribution) | labelled |
| ecommerce | `data/raw/ecommerce.xlsx` | Kaggle — [ankitverma2010/ecommerce-customer-churn-analysis-and-prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) | As listed on Kaggle page | labelled |
| subscription | `data/raw/subscription.csv` | Kaggle — [safrin03/predictive-analytics-for-customer-churn-dataset](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset) | As listed on Kaggle page | labelled |
| retail | `data/raw/retail.xlsx` | UCI ML Repository — [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | CC BY 4.0 | transactions |

All datasets are public. No real personal data is used beyond what these public sources publish.

## Notes per dataset
- **bank** — `Churn_Modelling.csv` renamed. 10,000 rows, churn rate 20.37%.
- **ecommerce** — `E Commerce Dataset.xlsx` renamed; the data is on sheet `E Comm` (sheet `Data Dict` is a data dictionary). 5,630 rows, churn rate 16.84%.
- **subscription** — the Kaggle download has `train.csv` (labelled) and `test.csv` (no `Churn` column, competition hold-out). Only `train.csv` is used, renamed to `subscription.csv`; `data_descriptions.csv` kept alongside as `subscription_descriptions.csv`. 243,787 rows, churn rate 18.12%.
- **retail** — `online_retail_II.xlsx` renamed. Two yearly sheets (2009-12-01 → 2011-12-09), 1,067,371 rows, 5,942 customers. ~23% of rows have no Customer ID; ~2% are cancellations (invoice starts with `C`, negative quantity); the two sheets overlap in early Dec 2010, creating duplicate rows.
