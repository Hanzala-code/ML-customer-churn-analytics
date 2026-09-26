# ML-Powered Customer Churn Analytics

Predicting which telecom customers will churn using classical ML models
trained on the IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn rate).

## What this project covers
- Leak-free preprocessing pipeline with stratified train/test split
- Baseline → Logistic Regression → Decision Tree → Random Forest
- Confusion matrix, precision, recall, F1 computed by hand and verified with sklearn
- ROC-AUC curve and cost-based threshold selection (PKR 6,000 vs PKR 1,000)
- Odds ratio interpretation of LR coefficients
- Overfitting analysis across Decision Tree depths
- Class imbalance handling with class_weight='balanced'
- Feature engineering evaluated against baseline AUC

## Key results
| Model | AUC | Recall (t=0.20) |
|---|---|---|
| Baseline (always Stay) | 0.500 | 0.000 |
| Logistic Regression | 0.840 | — |
| Decision Tree (d=5) | — | — |
| Random Forest | 0.842 | 0.856 |

**Chosen threshold: 0.20** — catches 108 more churners than default 0.5,
saving an estimated PKR 285,000 net over the default threshold.
