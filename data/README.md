# Threshold-specific datasets

This directory is intended to contain the three cleaned gamma-secretase descriptor datasets used by the current analysis:

- `Gama_secret_pub500.csv`
- `Gama_secret_pub1000.csv`
- `Gama_secret_pub2000.csv`

The three files contain the same molecules and the same descriptor columns. They differ only in the binary activity label that defines whether a compound is considered active:

| File | Activity column | Active-compound cutoff |
|---|---|---:|
| `Gama_secret_pub500.csv` | `Actividad_500` | IC50 <= 500 nM |
| `Gama_secret_pub1000.csv` | `Actividad_1000` | IC50 <= 1000 nM |
| `Gama_secret_pub2000.csv` | `Actividad_2000` | IC50 <= 2000 nM |

The local source files in `Mol_con_result` are:

- `v2_500/Gama_secret_pub500.csv`
- `v2_1000/Gama_secret_pub1000.csv`
- `v2_2000/Gama_secret_pub2000.csv`

Use the activity column matching the threshold selected for a given model run.
