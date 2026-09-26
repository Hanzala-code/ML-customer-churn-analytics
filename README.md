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
- Baseline (always "stay"): accuracy 0.735
- Best model: Random Forest, AUC 0.842, recall 0.856 at threshold 0.20
- Top churn drivers (permutation importance): tenure, TotalCharges, Contract_Two year
- Threshold chosen: 0.20, because a missed churner costs PKR 6,000 in lost revenue vs PKR 1,000 for an unnecessary retention offer — the 6:1 cost ratio makes aggressive flagging rational (t* = 0.14 by decision theory)
- Engineered features: n_services, is_new, charge_per_mo, price_jump; effect on AUC: 0.8422 → 0.8420 (no improvement — Random Forest already captures these interactions internally)
- Biggest lesson: accuracy is a misleading metric on imbalanced data — a model that catches zero churners can still score 73.5%, so always check recall and AUC before trusting any headline number


## Key results
| Model | AUC | Recall (t=0.20) |
|---|---|---|
| Baseline (always Stay) | 0.500 | 0.000 |
| Logistic Regression | 0.840 | — |
| Decision Tree (d=5) | — | — |
| Random Forest | 0.842 | 0.856 |

**Chosen threshold: 0.20** — catches 108 more churners than default 0.5,
saving an estimated PKR 285,000 net over the default threshold.
