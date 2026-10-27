# EDA — bank

Training rows: 8,000 · churn rate: 20.4%

## Findings

1. Overall, 20.4% of customers in the training data churned.
2. Customers with NumOfProducts = 3 churn at 82.1% vs 20.4% overall (4.0× more likely; n=207).
3. Churners have higher Age (median 45 vs 36 for stayers; medium difference, d=+0.73).
4. Customers with Geography = Germany churn at 33.1% vs 20.4% overall (1.6× more likely; n=2,005).
5. Churners have higher Balance (median 109,253 vs 91,715 for stayers; small difference, d=+0.31).

## Strongest numeric differences

| column | mean_churned | mean_stayed | median_churned | median_stayed | std_diff |
|---|---|---|---|---|---|
| Age | 44.781 | 37.454 | 45.0 | 36.0 | 0.733 |
| Balance | 91337.278 | 72554.144 | 109252.725 | 91715.195 | 0.311 |
| NumOfProducts | 1.467 | 1.546 | 1.0 | 2.0 | -0.117 |
| CreditScore | 644.261 | 652.414 | 645.0 | 654.0 | -0.083 |
| EstimatedSalary | 102026.976 | 99143.253 | 104721.985 | 98038.18 | 0.05 |
| RowNumber | 4930.181 | 5005.851 | 4882.5 | 5017.5 | -0.026 |
| Tenure | 5.003 | 5.02 | 5.0 | 5.0 | -0.006 |

## Groups furthest from the overall churn rate

| column | value | n | churn_rate | lift |
|---|---|---|---|---|
| NumOfProducts | 3 | 207 | 0.8213 | 4.031 |
| NumOfProducts | 2 | 3672 | 0.0738 | 0.362 |
| Geography | Germany | 2005 | 0.3307 | 1.623 |
| NumOfProducts | 1 | 4071 | 0.2798 | 1.373 |
| IsActiveMember | 0 | 3881 | 0.2669 | 1.31 |
| IsActiveMember | 1 | 4119 | 0.1442 | 0.708 |
| Gender | Female | 3613 | 0.2508 | 1.231 |
| Geography | France | 3993 | 0.159 | 0.781 |
| Gender | Male | 4387 | 0.165 | 0.81 |
| Geography | Spain | 2002 | 0.1658 | 0.814 |
