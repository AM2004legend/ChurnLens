# ChurnLens — 2-minute demo script

**Setup:** app open on *Demo: bank* (or *Synthetic telecom* on the deployed link). Train once before you start.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:15 | Title | "ChurnLens is one pipeline that predicts who will leave, explains why, and decides who is worth a retention offer — and it runs unchanged on banking, e-commerce, subscription and raw retail data." |
| 0:15–0:35 | **Leakage** tab | "Before any model is trained, a leakage guard checks the training split for ID-like columns, columns that predict churn suspiciously well, outcome-like names and dates. Here it flags RowNumber; I decide what to drop. On the e-commerce data it caught something subtler — two-thirds of rows were near-copies of other customers, which pushed test AUC to 0.998. Splitting by customer group brought it to an honest 0.94." |
| 0:35–0:50 | **EDA** tab | "Plain-English findings from the numbers: customers with 3 products churn at 82%, German customers at 33% vs 20% overall." |
| 0:50–1:05 | **Models** tab | "Logistic regression vs LightGBM, compared by 5-fold cross-validation on train only. I pick by PR-AUC, because a retention team cares about the top of the list. The test set is used once, at the end." |
| 1:05–1:30 | **Threshold** tab — drag the slider | "Instead of 0.5 I choose the threshold that maximises expected savings from the business's costs — offer cost, customer value, success rate, all labelled as assumptions. Targeting everyone *loses* money here; the chosen threshold earns the most. Change the costs and the recommendation moves." |
| 1:30–1:45 | **Explain** tab | "For any customer, SHAP gives the top reasons in plain words — for example 3 products and age 57 vs a typical 37. These show what the model uses, not proven causes." |
| 1:45–2:00 | **Segments** tab — click download | "At-risk customers are grouped by their main driver into segments with a suggested action, and the target list downloads as IDs only. For the retail data there was no churn label at all — I defined churn as 90 days without a purchase, built RFM features, and split by time so the model is judged on the future." |
