# ORVI-HCR: A Binned Regression Network for Fine-Grained ATP Prediction from Organoid Brightfield Images

## 🧬 Overview

**ORVI-HCR** is a deep learning framework designed to predict ATP levels from organoid brightfield microscopy images. It combines *interval classification* and *within-bin regression* in a two-phase training strategy to improve the accuracy and generalizability of ATP estimation, especially for challenging low-ATP and under-represented samples.

This model outperforms standard CNN baselines and supports explainable predictions via class-aware attention and heatmap visualization.

---

## 📁 Project Structure

```
ORVI-HCR/
│
├── data/
│   ├── processed_image_atp_mapping.json     # Mapping from image paths to ATP values
│   └── example_prediction.csv               # Example prediction outputs
│
├── generate_heatmap/                        # Heatmap generation for models
│
├── model/
│   ├── ATP_HCR.py                           # Main model architecture
│   ├── Benchmarks.py                        # Benchmarks model architecture
│   └── utils.py                             # Auxiliary functions
│
├── predict/                                 # batch predict and generate csv
│
├── train_phase1.py                          # Phase 1: training classification head
├── train_phase2.py                          # Phase 2: training regression heads
├── EXP_train_AD_continue.py                 # Training script for baseline in continue style
├── EXP_train_MutiHead_BenchMark.py          #  train in BenchMark
│
├── evaluate_model_metrics.py                # Evaluation script (MAE, MAPE, ACC, R²)
├── requirements.txt                         # Python environment dependencies
└── README.md                                # Project documentation
```

---

## 🔧 Environment Setup

Requires Python ≥ 3.8. We recommend using a virtual environment:

```bash
conda create -n atp_env python=3.8
conda activate atp_env
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1️⃣ Train ORVI-HCR

**Phase 1: Classification training**

```bash
python train_phase1.py
```

**Phase 2: Regression training (freeze classifier)**

```bash
python train_phase2.py
```

### 2️⃣ Predict ATP values

```bash
python predict/pred_csv.py
```

### 3️⃣ Generate Heatmaps

```bash
python generate_heatmap/pred_csv_MutiQ_save.py
```

---

## 📈 Evaluation

Evaluate prediction performance with:

```bash
python evaluate_model_metrics.py
```

Reported metrics include:

* MAE (Mean Absolute Error)
* MAPE (Mean Absolute Percentage Error)
* Pair ACC (Pairwise Accuracy for ranking)
* R² Score (Coefficient of Determination)

## 📌 Model Design: ORVI-HCR

ORVI-HCR includes:

1. **Feature Extractor**: Deep CNN with multi-scale filters
2. **Attention Module**: Learns spatially focused representations
3. **Classifier Head**: Outputs ATP bin probabilities
4. **Multiple Regression Heads**: Predict normalized position within each ATP bin
5. **Final ATP Estimation**

---

## 📖 Citation

If you find **ORVI-HCR** useful in your research, please consider citing our work:

```bibtex
# Citation placeholder in future
```

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or contact the authors.
