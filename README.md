## Week 2: Building ML Models
- Baseline (always "stay"): accuracy 0.735
- Best model: Random Forest, AUC 0.842, recall 0.856 at threshold 0.20
- Top churn drivers (permutation importance): tenure, TotalCharges, Contract_Two year
- Threshold chosen: 0.20, because a missed churner costs PKR 6,000 in lost revenue vs PKR 1,000 for an unnecessary retention offer — the 6:1 cost ratio makes aggressive flagging rational (t* = 0.14 by decision theory)
- Engineered features: n_services, is_new, charge_per_mo, price_jump; effect on AUC: 0.8422 → 0.8420 (no improvement — Random Forest already captures these interactions internally)
- Biggest lesson: accuracy is a misleading metric on imbalanced data — a model that catches zero churners can still score 73.5%, so always check recall and AUC before trusting any headline number.
