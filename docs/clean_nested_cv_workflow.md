# Clean analysis workflow

This document records the cleaned procedure produced in `Mol_con_result` for the gamma-secretase ML screening project.

## Main principle

Performance estimates are reported only from held-out outer folds of nested cross-validation. Final full-data pipelines are kept for screening and exploratory interpretation, but they are not used to estimate model performance.

## Inputs

The workflow expects one result directory per IC50 threshold:

- `v2_500/`
- `v2_1000/`
- `v2_2000/`

Each threshold directory contains model-specific run outputs for logistic regression, NCA+KNN, SVM, random forest, Gaussian process classifier, MLP, and XGBoost.

## Clean procedure

1. Inventory all available runs and expected output files.
2. Select one canonical complete run per threshold/model without using performance metrics for selection.
3. Exclude duplicate or non-canonical XGBoost runs from the main analysis.
4. Build a long-form table from `*_outer_fold_metrics.csv` only.
5. Generate summary tables and figures from outer-fold metrics.
6. Audit leakage risks and verify that final `.joblib` models do not feed the performance tables.
7. Run paired Friedman tests within each IC50 threshold and metric, using outer folds as paired blocks.
8. When global tests are significant, run Wilcoxon signed-rank post-hoc tests with Holm correction.
9. Run fixed-hyperparameter repeated-CV analysis only as a supplementary stability check.
10. Restrict interpretability outputs to canonical manifest runs and label them exploratory.

## Canonical commands

Run from the cleaned result folder:

```bash
python analyze_nested_cv_results.py --base_dir . --outdir analysis_outputs
python final_statistical_model_comparison.py --metrics analysis_outputs/tables/all_outer_fold_metrics_long.csv --outdir analysis_outputs/final_statistics
python fixed_hyperparameter_stability_analysis.py --base_dir . --analysis_outdir analysis_outputs --outdir analysis_outputs/fixed_hparam_stability
python interpretability_analysis.py --base_dir . --outdir analysis_outputs --use_manifest
```

## Main outputs

- `analysis_outputs/tables/included_runs_manifest.csv`
- `analysis_outputs/tables/excluded_runs_manifest.csv`
- `analysis_outputs/tables/all_outer_fold_metrics_long.csv`
- `analysis_outputs/tables/summary_by_model_threshold.csv`
- `analysis_outputs/tables/global_ranking_complete_only.csv`
- `analysis_outputs/canonical_selection_log.md`
- `analysis_outputs/leakage_audit.md`
- `analysis_outputs/final_statistics/reports/final_statistical_comparison_report.md`
- `analysis_outputs/fixed_hparam_stability/reports/fixed_hparam_stability_report.md`

## Interpretation rules

- Do not describe any model as significantly superior unless the specific Holm-corrected post-hoc comparison supports that claim.
- Prefer descriptive language such as `performed best under the evaluated protocol` when ranking by mean metrics.
- Treat the fixed-hyperparameter repeated-CV results as stability evidence, not as independent external validation.
- Treat final full-data models as deployment/screening artifacts only.
