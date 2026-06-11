# Leakage audit

The audited training scripts write the primary metric files from held-out outer-fold predictions. Final full-data pipelines are fitted after nested-CV summaries and are not read by this analysis script for performance.

- Pipeline with preprocessing/model inside CV: PASS
- SMOTE inside imblearn Pipeline: PASS
- Stratified outer CV: PASS
- Inner CV for hyperparameter selection: PASS
- Inner CV for decision-threshold selection: PASS
- Metrics derived from outer folds: PASS
- Final joblib used only after evaluation: PASS

## predict_proba(X) classification

- Classification: unclear_manual_review_required
- Main-metric impact: this analysis reads only canonical `*_outer_fold_metrics.csv` files. Any `predict_proba(X)` use in final full-data screening or interpretability is harmless for reported performance, but would be dangerous if its outputs were copied into main performance tables.

Conclusion: No evidence that final `.joblib` predictions feed the main performance tables; outer-fold-only reporting is enforced by the cleaned analysis script.
