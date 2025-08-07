# Virtual Screening of Gamma-Secretase Inhibitors using Machine Learning

This repository contains the code and data used for virtual screening of compounds targeting **gamma-secretase**, an enzyme associated with the pathogenesis of Alzheimer’s disease.

Several machine learning models were trained and optimized using **Optuna** to classify active compounds based on physicochemical descriptors. The results were part of the research article:

> **"Evaluating Machine Learning Algorithms for Classifying Active Compounds in Alzheimer’s Disease" (2025)**

## 🧠 Project Structure

```bash
gamma-secretase-ml-screening/
├── data/                  # Dataset used for training/testing
├── scripts/               # Individual ML model training scripts (ANN, RF, SVM, etc.)
├── evaluation/            # Model evaluation and comparison
├── results/               # Output metrics, graphs or reports
├── README.md              # Project overview and setup instructions
├── requirements.txt       # Python dependencies
└── LICENSE                # Project license (MIT)
```

## 💻 Infrastructure

All model training and evaluation were conducted on **FinisTerrae III**, the flagship supercomputer at CESGA, which provides both high-performance CPU and GPU resources interconnected by a low-latency network.  

<img width="2171" height="1440" alt="FT3_schema" src="https://github.com/user-attachments/assets/7bea0591-faf8-4736-80e3-e8a7051867b4" />

- **Node used for benchmarking (“ilk”):**  
  - 256 compute nodes  
  - 2× Intel Xeon Ice Lake 8352Y (32 cores each → 64 cores/node) → 16,384 cores total  
  - 256 GB RAM (247 GB usable)  
  - 960 GB SSD NVMe local storage  
  - Infiniband HDR 100 network connection  

For more details, see the CESGA user guide:  
<https://cesga-docs.gitlab.io/ft3-user-guide/overview.html>

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```
## 📊 Dataset Information

- **Source:** ChEMBL (CHEMBL2094135)  
- **File:** `data/Gama_secret_pub.csv`  
- **Entries:** 1,745 compounds  
- **Activity labels:**  
  - **Actives** (IC₅₀ < 10 nM): 1,367  
  - **Inactives:** 378  
- **Preprocessing:**  
  - SMILES standardization & deduplication performed prior to descriptor calculation  
- **Descriptor calculation:**  
  - ~200 physicochemical & topological descriptors computed with **AlvaMolecule v2.0.10**
## 💻 Code Information

- **`data/`**:  
  - Raw (`raw/`) and processed (`processed/`) CSVs  
- **`scripts/`**:  
  - `train_ann.py` – Multi-Layer Perceptron (MLP) training  
  - `train_rf.py` – Random Forest training  
  - `train_svm.py` – Support Vector Machine training  
  - `train_gp.py` – Gaussian Process Classifier  
  - `train_knn_nca.py` – KNN with Neighborhood Component Analysis  
  - `train_lr.py` – Logistic Regression  
- **`evaluation/`**:  
  - `eval_mods.py` – 10-fold cross-validation and metric aggregation  
  - `plot_stats.py` – Statistical tests (ANOVA, Tukey HSD) and plotting  

## ⚙️ Methodology

1. **Data curation**  
   - Clean and standardize SMILES; remove duplicates  
2. **Descriptor calculation**  
   - Compute ~200 physicochemical and topological descriptors with AlvaMolecule v2.0.10  
3. **Preprocessing**  
   - Normalize features (StandardScaler)  
   - Reduce dimensionality via PCA (retain 30 components)  
4. **Model training**  
   - Six algorithms (MLP, RF, SVM, GP, KNN+NCA, LR)  
   - Hyperparameter optimization with Optuna TPE (100–300 trials per model)  
   - Validation: 10-fold cross-validation, F1-macro as primary metric  
5. **Statistical analysis**  
   - Normality test: Shapiro–Wilk  
   - Homoscedasticity: Bartlett’s test  
   - ANOVA + Tukey HSD for pairwise model comparisons

## 🚀 How to Run

1. Place your dataset in the `data/` folder (e.g., `Gama_secretM.csv`).
2. Run any script in the `scripts/` folder to train models individually.
3. Use `eval_mods.py` in `evaluation/` to compare models with cross-validation metrics.
4. Results will be saved in `.csv` files inside `results/`.

## 📚 Citation

> Domínguez Gortaire, J. A. (2025). *Evaluating Machine Learning Algorithms for Classifying Active Compounds in Alzheimer’s Disease.*

## 📄 License

MIT License
