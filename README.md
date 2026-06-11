# Virtual Screening of Gamma-Secretase Inhibitors using Machine Learning

This repository contains code, data documentation, and cleaned analysis outputs for machine-learning classification of gamma-secretase inhibitors associated with Alzheimer's disease research.

The project evaluates multiple supervised classifiers for active/inactive compound classification across IC50 activity thresholds. The cleaned workflow in `Mol_con_result` updates the original analysis by enforcing outer-fold-only performance reporting from nested cross-validation and by replacing the older ANOVA/Tukey interpretation with paired non-parametric model comparisons.

## Current Clean Workflow

The current cleaned procedure is documented in:

- `docs/clean_nested_cv_workflow.md`
- `results/clean_analysis/executive_summary.md`
- `results/clean_analysis/canonical_selection_log.md`
- `results/clean_analysis/leakage_audit.md`
- `results/clean_analysis/final_statistical_comparison_report.md`
- `results/clean_analysis/fixed_hparam_stability_report.md`
- `results/clean_analysis/global_ranking_complete_only.csv`

Core methodological rules:

1. Report performance only from held-out outer folds of nested cross-validation.
2. Select canonical runs without using performance metrics.
3. Exclude duplicate and non-canonical runs from the main analysis.
4. Use final full-data `.joblib` pipelines only for later screening or exploratory interpretation, not for performance estimation.
5. Compare classifiers with Friedman tests over paired outer folds, followed by Wilcoxon signed-rank tests with Holm correction when appropriate.
6. Treat fixed-hyperparameter repeated-CV results as supplementary stability evidence, not external validation.

## Data

The `data/` directory contains three threshold-specific datasets:

- `data/Gama_secret_pub500.csv`
- `data/Gama_secret_pub1000.csv`
- `data/Gama_secret_pub2000.csv`

The three files contain the same molecules and the same molecular descriptors. They differ only in the binary activity label used to define active compounds:

| File | Activity column | Active-compound cutoff |
|---|---|---:|
| `Gama_secret_pub500.csv` | `Actividad_500` | IC50 <= 500 nM |
| `Gama_secret_pub1000.csv` | `Actividad_1000` | IC50 <= 1000 nM |
| `Gama_secret_pub2000.csv` | `Actividad_2000` | IC50 <= 2000 nM |

This allows the same descriptor matrix to be evaluated under three increasingly permissive activity definitions.

## Models

The analysis covers the following classifiers:

- Logistic regression (`logreg`)
- Neighborhood Components Analysis + KNN (`nca_knn`)
- Support Vector Machine (`svm`)
- Random Forest (`rf`)
- Gaussian Process Classifier (`gpc`)
- Multi-Layer Perceptron (`mlp`)
- XGBoost (`xgboost`)

## Thresholds

The cleaned analysis uses three IC50 thresholds:

- 500 nM
- 1000 nM
- 2000 nM

## Clean Analysis Summary

The cleaned nested-CV analysis reports:

- Performance source: outer-fold nested-CV metrics only
- Final pipelines used for performance: false
- Duplicate runs excluded: 15
- Top global complete model by descriptive ranking: random forest

Across complete canonical runs and primary metrics, the descriptive global ranking was:

| Rank | Model | Mean macro-F1 | Mean PR-AUC | Mean ROC-AUC |
|---:|---|---:|---:|---:|
| 1 | rf | 0.7357 | 0.9290 | 0.8310 |
| 2 | nca_knn | 0.7291 | 0.9064 | 0.8015 |
| 3 | xgboost | 0.7290 | 0.9265 | 0.8242 |
| 4 | mlp | 0.7275 | 0.9256 | 0.8181 |
| 5 | svm | 0.7249 | 0.9263 | 0.8200 |
| 6 | gpc | 0.7248 | 0.9244 | 0.8164 |
| 7 | logreg | 0.6779 | 0.8973 | 0.7612 |

No Wilcoxon pairwise comparison reached Holm-adjusted significance in the final paired statistical comparison. Interpret descriptive rankings with caution because the primary nested-CV comparison has five paired outer folds per threshold.

## Reproducibility Commands

From the cleaned result directory used to generate the current outputs:

```bash
python analyze_nested_cv_results.py --base_dir . --outdir analysis_outputs
python final_statistical_model_comparison.py --metrics analysis_outputs/tables/all_outer_fold_metrics_long.csv --outdir analysis_outputs/final_statistics
python fixed_hyperparameter_stability_analysis.py --base_dir . --analysis_outdir analysis_outputs --outdir analysis_outputs/fixed_hparam_stability
python interpretability_analysis.py --base_dir . --outdir analysis_outputs --use_manifest
```

## Data Information

- Source: ChEMBL (CHEMBL2094135)
- Descriptor calculation: approximately 200 physicochemical and topological descriptors computed with AlvaMolecule v2.0.10
- Preprocessing: SMILES standardization, deduplication, scaling, PCA, and class-balancing procedures as implemented in the analysis scripts

## Infrastructure

Model training and evaluation were conducted on FinisTerrae III at CESGA.

Benchmarking node context from the original project documentation:

- 2 x Intel Xeon Ice Lake 8352Y processors per node
- 64 cores per node
- 256 GB RAM per node
- NVMe local storage
- Infiniband HDR 100 network connection

CESGA user guide: <https://cesga-docs.gitlab.io/ft3-user-guide/overview.html>

## Requirements

Install the Python dependencies before rerunning the workflow:

```bash
pip install -r requirements.txt
```

The cleaned scripts require the scientific Python stack used by the project, including pandas, numpy, scipy, scikit-learn, matplotlib, imbalanced-learn, joblib, and xgboost when XGBoost models are included.

## Citation

Dominguez Gortaire, J. A. (2025). *Evaluating Machine Learning Algorithms for Classifying Active Compounds in Alzheimer's Disease.*

## License

MIT License
