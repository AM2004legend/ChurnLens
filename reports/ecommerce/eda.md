# EDA — ecommerce

Training rows: 4,503 · churn rate: 16.8%

## Findings

1. Overall, 16.8% of customers in the training data churned.
2. Customers with NumberOfDeviceRegistered = 6 churn at 35.6% vs 16.8% overall (2.1× more likely; n=132).
3. Churners have lower Tenure (median 1 vs 10 for stayers; large difference, d=-1.17).
4. Customers with Complain = 1 churn at 31.1% vs 16.8% overall (1.8× more likely; n=1,280).
5. Churners have lower CashbackAmount (median 150 vs 165 for stayers; small difference, d=-0.45).

## Strongest numeric differences

| column | mean_churned | mean_stayed | median_churned | median_stayed | std_diff |
|---|---|---|---|---|---|
| Tenure | 3.278 | 11.547 | 1.0 | 10.0 | -1.166 |
| CashbackAmount | 159.784 | 179.858 | 149.62 | 165.45 | -0.453 |
| DaySinceLastOrder | 3.211 | 4.775 | 2.0 | 3.0 | -0.434 |
| SatisfactionScore | 3.417 | 2.999 | 3.0 | 3.0 | 0.309 |
| NumberOfDeviceRegistered | 3.949 | 3.642 | 4.0 | 4.0 | 0.301 |
| WarehouseToHome | 17.26 | 15.401 | 15.0 | 13.0 | 0.215 |
| CityTier | 1.781 | 1.607 | 1.0 | 1.0 | 0.188 |
| NumberOfAddress | 4.538 | 4.137 | 3.0 | 3.0 | 0.151 |

## Groups furthest from the overall churn rate

| column | value | n | churn_rate | lift |
|---|---|---|---|---|
| NumberOfDeviceRegistered | 6 | 132 | 0.3561 | 2.115 |
| Complain | 1 | 1280 | 0.3109 | 1.847 |
| PreferedOrderCat | Grocery | 331 | 0.0483 | 0.287 |
| MaritalStatus | Single | 1423 | 0.2741 | 1.628 |
| PreferedOrderCat | Mobile Phone | 1683 | 0.2709 | 1.61 |
| PreferedOrderCat | Others | 205 | 0.0683 | 0.406 |
| NumberOfDeviceRegistered | 1 | 187 | 0.0856 | 0.508 |
| PreferredPaymentMode | Cash on Delivery | 403 | 0.2506 | 1.489 |
| SatisfactionScore | 5 | 882 | 0.2494 | 1.482 |
| PreferedOrderCat | Laptop & Accessory | 1636 | 0.1039 | 0.617 |
