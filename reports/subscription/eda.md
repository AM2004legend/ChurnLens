# EDA — subscription

Training rows: 195,029 · churn rate: 18.1%

## Findings

1. Overall, 18.1% of customers in the training data churned.
2. Customers with SupportTicketsPerMonth = 9 churn at 23.7% vs 18.1% overall (1.3× more likely; n=19,486).
3. Churners have lower AccountAge (median 40 vs 65 for stayers; medium difference, d=-0.53).
4. Customers with PaymentMethod = Credit card churn at 16.2% vs 18.1% overall (1.1× less likely; n=48,674).
5. Churners have lower AverageViewingDuration (median 70 vs 97.5 for stayers; small difference, d=-0.39).

## Strongest numeric differences

| column | mean_churned | mean_stayed | median_churned | median_stayed | std_diff |
|---|---|---|---|---|---|
| AccountAge | 45.741 | 63.172 | 40.0 | 65.0 | -0.528 |
| AverageViewingDuration | 76.552 | 95.795 | 70.036 | 97.503 | -0.39 |
| ContentDownloadsPerMonth | 20.566 | 25.364 | 19.0 | 26.0 | -0.338 |
| ViewingHoursPerWeek | 17.469 | 21.199 | 16.135 | 21.565 | -0.337 |
| TotalCharges | 617.79 | 779.355 | 489.355 | 683.6 | -0.316 |
| MonthlyCharges | 13.409 | 12.288 | 13.863 | 12.191 | 0.261 |
| SupportTicketsPerMonth | 5.027 | 4.388 | 5.0 | 4.0 | 0.224 |
| UserRating | 3.061 | 2.991 | 3.091 | 2.982 | 0.061 |

## Groups furthest from the overall churn rate

| column | value | n | churn_rate | lift |
|---|---|---|---|---|
| SupportTicketsPerMonth | 9 | 19486 | 0.237 | 1.308 |
| SupportTicketsPerMonth | 0 | 19387 | 0.1329 | 0.733 |
| SupportTicketsPerMonth | 8 | 19498 | 0.2255 | 1.244 |
| SupportTicketsPerMonth | 1 | 19407 | 0.1446 | 0.798 |
| SupportTicketsPerMonth | 2 | 19591 | 0.1513 | 0.835 |
| SupportTicketsPerMonth | 7 | 19740 | 0.207 | 1.142 |
| SupportTicketsPerMonth | 3 | 19452 | 0.1597 | 0.881 |
| PaymentMethod | Credit card | 48674 | 0.1619 | 0.893 |
| SubscriptionType | Premium | 64615 | 0.1633 | 0.901 |
| GenrePreference | Action | 39156 | 0.1647 | 0.909 |
