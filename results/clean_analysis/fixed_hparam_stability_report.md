# Fixed-hyperparameter supplementary stability analysis

This analysis is supplementary. It evaluates stability and consistency of models conditioned on hyperparameters selected previously; it is not an unbiased estimate of hyperparameter selection and does not replace the primary nested cross-validation analysis.

The repeated stratified CV design increases the number of paired blocks used for descriptive stability comparisons, but the 20 blocks are repeated splits of the same datasets and must not be interpreted as validation on 20 fully independent samples or as external validation.

Final full-data models were trained only as artifacts for later screening and manuscript follow-up work, not for performance reporting.

## Best mean macro-F1 by threshold

- 500 nM: rf (mean macro-F1=0.737)
- 1000 nM: rf (mean macro-F1=0.737)
- 2000 nM: rf (mean macro-F1=0.738)

## Statistical caution

Friedman and Wilcoxon tests use repeated-CV blocks as paired observations. Because those partitions are dependent, statistical comparisons should be interpreted cautiously as stability/consistency checks. In this supplementary analysis, 59 model-vs-XGBoost Holm-adjusted comparisons were significant.

## Ranking stability

Global descriptive ranking across primary metrics:

| model_raw | mean_rank | median_rank | top_rank_count |
|---|---:|---:|---:|
| rf | 1.66667 | 1 | 5 |
| gpc | 2.44444 | 2 | 2 |
| svm | 3.22222 | 4 | 2 |
| xgboost | 3.66667 | 3 | 0 |
| mlp | 4.33333 | 5 | 0 |
| nca_knn | 5.66667 | 6 | 0 |
| logreg | 7.00000 | 7 | 0 |

## Differences against XGBoost

The complete local output table is `analysis_outputs/fixed_hparam_stability/tables/fixed_hparam_model_vs_xgboost.csv`. Mean deltas are also available locally in `analysis_outputs/fixed_hparam_stability/tables/fixed_hparam_summary_by_model_threshold.csv`.

## Final model registry

Final full-data model artifacts were generated locally under `analysis_outputs/fixed_hparam_stability/final_models/`. They are intentionally not uploaded here as primary performance evidence. Use them only as screening artifacts and keep the corresponding threshold, parameter, feature-name, and model-card files with each model when transferring them.
