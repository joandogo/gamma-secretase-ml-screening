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

## 🚀 How to Run

1. Place your dataset in the `data/` folder (e.g., `Gama_secretM.csv`).
2. Run any script in the `scripts/` folder to train models individually.
3. Use `eval_mods.py` in `evaluation/` to compare models with cross-validation metrics.
4. Results will be saved in `.csv` files inside `results/`.

## 📚 Citation

> Domínguez Gortaire, J. A. (2025). *Evaluating Machine Learning Algorithms for Classifying Active Compounds in Alzheimer’s Disease.*

## 📄 License

MIT License
