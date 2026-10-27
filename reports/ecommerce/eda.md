# EDA — ecommerce

Training rows: 4,504 · churn rate: 16.8%

## Findings

1. Overall, 16.8% of customers in the training data churned.
2. Customers with NumberOfDeviceRegistered = 6 churn at 37.6% vs 16.8% overall (2.2× more likely; n=133).
3. Churners have lower Tenure (median 1 vs 10 for stayers; large difference, d=-1.16).
4. Customers with Complain = 1 churn at 31.7% vs 16.8% overall (1.9× more likely; n=1,262).
5. Churners have lower CashbackAmount (median 150 vs 167 for stayers; small difference, d=-0.48).

## Strongest numeric differences

| column | mean_churned | mean_stayed | median_churned | median_stayed | std_diff |
|---|---|---|---|---|---|
| Tenure | 3.317 | 11.512 | 1.0 | 10.0 | -1.16 |
| CashbackAmount | 159.654 | 180.973 | 149.51 | 166.52 | -0.484 |
| DaySinceLastOrder | 3.187 | 4.812 | 2.0 | 4.0 | -0.462 |
| NumberOfDeviceRegistered | 3.96 | 3.643 | 4.0 | 4.0 | 0.312 |
| SatisfactionScore | 3.41 | 2.998 | 3.0 | 3.0 | 0.303 |
| CityTier | 1.863 | 1.625 | 1.0 | 1.0 | 0.254 |
| WarehouseToHome | 17.161 | 15.444 | 15.0 | 13.0 | 0.202 |
| NumberOfAddress | 4.441 | 4.148 | 3.0 | 3.0 | 0.111 |

## Groups furthest from the overall churn rate

| column | value | n | churn_rate | lift |
|---|---|---|---|---|
| NumberOfDeviceRegistered | 6 | 133 | 0.3759 | 2.234 |
| Complain | 1 | 1262 | 0.317 | 1.883 |
| PreferedOrderCat | Grocery | 330 | 0.0394 | 0.234 |
| PreferedOrderCat | Others | 205 | 0.0537 | 0.319 |
| PreferedOrderCat | Mobile Phone | 1651 | 0.275 | 1.634 |
| MaritalStatus | Single | 1427 | 0.2663 | 1.582 |
| PreferredPaymentMode | Cash on Delivery | 403 | 0.2531 | 1.504 |
| NumberOfDeviceRegistered | 1 | 194 | 0.0876 | 0.521 |
| NumberOfDeviceRegistered | 2 | 216 | 0.088 | 0.523 |
| SatisfactionScore | 5 | 880 | 0.2432 | 1.445 |
