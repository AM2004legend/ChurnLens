# Leakage report — bank

Computed on the training split only.

| Column | Check | Value | Dropped in config? | Why flagged |
|---|---|---|---|---|
| RowNumber | id_like | 1.0 | yes | 100% unique values — looks like an identifier; a model can memorise it but it carries no general signal. |

## Single-feature ROC-AUC (max(AUC, 1−AUC))

| Column | AUC |
|---|---|
| Age | 0.7283 |
| IsActiveMember | 0.5945 |
| Geography | 0.5905 |
| NumOfProducts | 0.5836 |
| Balance | 0.5792 |
| Gender | 0.5551 |
| Surname | 0.5246 |
| CreditScore | 0.521 |
| EstimatedSalary | 0.5146 |
| HasCrCard | 0.5082 |
| RowNumber | 0.5075 |
| Tenure | 0.5017 |
