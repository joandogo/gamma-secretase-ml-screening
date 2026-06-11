# Final statistical model comparison

Friedman tests with folds as paired blocks replace ANOVA/Tukey because every classifier was evaluated on the same held-out outer folds within each IC50 threshold. ANOVA/Tukey assumes an analysis structure that is not the primary design here and is considered obsolete for the main results.

Metrics come exclusively from outer-fold nested-CV outputs; final `.joblib` pipelines were not loaded, predictions were not recalculated, and no raw result folders were modified.

- Alpha: 0.05
- Thresholds: 500, 1000, 2000
- Models: logreg, nca_knn, svm, rf, gpc, mlp, xgboost
- Folds by threshold: 5 outer folds per threshold
- Paired design valid: true

## Kendall's W interpretation

Kendall's W close to 0 suggests weak differences among model ranks, intermediate values suggest moderate separation, and high values suggest stronger rank consistency. Rigid cutoffs are avoided because only five outer folds are available.

## Friedman results

- 500 nM, f1_macro: chi-square=14.657, p=0.023, W=0.489.
- 500 nM, roc_auc: chi-square=20.743, p=0.002, W=0.691.
- 500 nM, pr_auc: chi-square=22.629, p<0.001, W=0.754.
- 500 nM, accuracy: chi-square=13.949, p=0.030, W=0.465.
- 500 nM, precision_macro: chi-square=13.714, p=0.033, W=0.457.
- 500 nM, recall_macro: chi-square=14.743, p=0.022, W=0.491.
- 500 nM, brier: chi-square=20.914, p=0.002, W=0.697.
- 1000 nM, f1_macro: chi-square=14.817, p=0.022, W=0.494.
- 1000 nM, roc_auc: chi-square=20.143, p=0.003, W=0.671.
- 1000 nM, pr_auc: chi-square=19.886, p=0.003, W=0.663.
- 1000 nM, accuracy: chi-square=19.261, p=0.004, W=0.642.
- 1000 nM, precision_macro: chi-square=16.323, p=0.012, W=0.544.
- 1000 nM, recall_macro: chi-square=17.527, p=0.008, W=0.584.
- 1000 nM, brier: chi-square=28.800, p<0.001, W=0.960.
- 2000 nM, f1_macro: chi-square=11.400, p=0.077, W=0.380.
- 2000 nM, roc_auc: chi-square=19.200, p=0.004, W=0.640.
- 2000 nM, pr_auc: chi-square=18.257, p=0.006, W=0.609.
- 2000 nM, accuracy: chi-square=16.022, p=0.014, W=0.534.
- 2000 nM, precision_macro: chi-square=16.629, p=0.011, W=0.554.
- 2000 nM, recall_macro: chi-square=10.301, p=0.113, W=0.343.
- 2000 nM, brier: chi-square=24.600, p<0.001, W=0.820.

## Holm-significant post-hoc comparisons

No Wilcoxon pairwise comparison reached Holm-adjusted significance.

## XGBoost-focused comparison

No model differed significantly from XGBoost after Holm correction within threshold-metric families.

## Statistical caution

Power is low with n=5 paired outer folds. P-values should be interpreted together with effect size, direction, calibration, and consistency across thresholds.

## Recommended article language

- Avoid `significantly superior` unless a Holm-corrected post-hoc test supports that exact comparison.
- Prefer `performed best under the evaluated protocol` for descriptive rankings.
- Use `differences were modest` when effect sizes or deltas are small.
- Do not infer biochemical mechanisms from model rankings.

## Suggested Methods/statistical analysis text

For each IC50 threshold and metric, classifier performance was compared across the same held-out outer folds using the Friedman test. Kendall's W was reported as an effect-size measure. When the Friedman test was significant, pairwise Wilcoxon signed-rank tests were performed with Holm correction. Brier score was interpreted with lower values indicating better calibration.
