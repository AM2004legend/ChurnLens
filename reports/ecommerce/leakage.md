# Leakage report — ecommerce

Computed on the training split only.

| Column | Check | Value | Dropped in config? | Why flagged |
|---|---|---|---|---|
| (all features) | duplicate_profiles | 0.0984 | grouped split | 9.8% of rows repeat another row's full feature profile under a different id — likely copies of the same customer. Use split.group_by so copies stay on one side of the split. |

## Single-feature ROC-AUC (max(AUC, 1−AUC))

| Column | AUC |
|---|---|
| Tenure | 0.801 |
| PreferedOrderCat | 0.6524 |
| Complain | 0.6448 |
| DaySinceLastOrder | 0.6374 |
| CashbackAmount | 0.6329 |
| MaritalStatus | 0.6242 |
| SatisfactionScore | 0.5838 |
| NumberOfDeviceRegistered | 0.5775 |
| WarehouseToHome | 0.565 |
| PreferredPaymentMode | 0.5567 |
| CityTier | 0.5461 |
| NumberOfAddress | 0.531 |
| PreferredLoginDevice | 0.5245 |
| OrderCount | 0.5228 |
| Gender | 0.514 |
| HourSpendOnApp | 0.514 |
| OrderAmountHikeFromlastYear | 0.5055 |
| CouponUsed | 0.5015 |
