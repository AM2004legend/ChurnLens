# EDA — retail

Training rows: 4,332 · churn rate: 64.1%

## Findings

1. Overall, 64.1% of customers in the training data churned.
2. Churners have higher recency_days (median 187 vs 38 for stayers; large difference, d=+0.95).
3. Churners have lower purchases_last_90 (median 0 vs 1 for stayers; medium difference, d=-0.69).
4. Churners have lower frequency (median 2 vs 4 for stayers; medium difference, d=-0.61).
5. Churners have lower purchases_prev_90 (median 0 vs 1 for stayers; medium difference, d=-0.59).

## Strongest numeric differences

| column | mean_churned | mean_stayed | median_churned | median_stayed | std_diff |
|---|---|---|---|---|---|
| recency_days | 161.804 | 75.123 | 187.0 | 38.0 | 0.946 |
| purchases_last_90 | 0.429 | 1.84 | 0.0 | 1.0 | -0.688 |
| frequency | 2.367 | 7.31 | 2.0 | 4.0 | -0.613 |
| purchases_prev_90 | 0.32 | 1.282 | 0.0 | 1.0 | -0.587 |
| tenure_days | 329.846 | 386.703 | 325.0 | 444.0 | -0.353 |
| monetary_total | 889.387 | 3680.04 | 459.395 | 1535.255 | -0.319 |
| purchase_trend | 0.109 | 0.559 | 0.0 | 0.0 | -0.305 |
| monetary_avg | 367.467 | 419.805 | 257.832 | 324.467 | -0.074 |

## Groups furthest from the overall churn rate

_(none)_
