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

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. Place your dataset in the `data/` folder (e.g., `Gama_secretM.csv`).
2. Run any script in the `scripts/` folder to train models individually.
3. Use `eval_mods.py` in `evaluation/` to compare models with cross-validation metrics.
4. Results will be saved in `.csv` files inside `results/`.

## 📚 Citation

> Domínguez Gortaire, J. A. (2025). *Evaluating Machine Learning Algorithms for Classifying Active Compounds in Alzheimer’s Disease.*

## 📄 License

MIT License
