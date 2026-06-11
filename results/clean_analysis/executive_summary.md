# Clean analysis executive summary

Source folder: `Mol_con_result`

- Analysis: canonical nested-CV analysis
- Analysis final: true
- Performance source: outer-fold nested-CV metrics only
- Final pipelines used for performance: false
- Duplicate runs excluded: 15
- Top global complete model by descriptive ranking: random forest

## Key methodological update

The cleaned workflow replaces the older simple cross-validation / ANOVA-style comparison with a nested-CV-first analysis and paired non-parametric statistical testing. This avoids using final full-data models for performance reporting and keeps the final pipelines restricted to screening or exploratory follow-up.

## Primary ranking summary

Across complete canonical runs and primary metrics, random forest ranked first descriptively, followed by NCA+KNN, XGBoost, MLP, SVM, GPC, and logistic regression. These rankings should be read together with the paired statistical tests and the caution that only five outer folds are available per threshold.
