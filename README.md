# 🤖 ML Classification Benchmark & Prediction Suite

> Six classical classifiers evaluated head-to-head across two real-world datasets, with a Streamlit app for live inference.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📂 Project Structure

```
Classification_Assignments2/
├── notebooks/              # Jupyter notebooks for EDA & training
│   ├── 01_titanic_classification.ipynb
│   └── 02_breast_cancer_classification.ipynb
├── python_files/           # Standalone training scripts
│   ├── titanic_model.py
│   └── breast_cancer_model.py
├── models/                 # Serialized model & scaler artifacts (.pkl)
│   ├── titanic_best_model.pkl
│   ├── titanic_scaler.pkl
│   ├── breast_cancer_best_model.pkl
│   └── breast_cancer_scaler.pkl
├── datasets/               # Raw training data
│   └── train.csv
├── app/                    # Streamlit inference app
│   └── app.py
├── screenshots/            # Output visualizations
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

### 🚢 Titanic Survival
- **Source:** Seaborn built-in dataset
- **Task:** Binary classification — survived (1) vs not survived (0)
- **Features used:** `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`
- **Preprocessing:** Label encoding for categorical columns, `StandardScaler` for distance-based models

### 🧬 Breast Cancer
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Task:** Binary classification — Malignant (0) vs Benign (1)
- **Features:** 30 computed cell nucleus measurements
- **Preprocessing:** `StandardScaler` for distance-based models

---

## 🏆 Benchmark Results

### Titanic Dataset (80/20 split, `random_state=42`)

| Algorithm | Accuracy | Scaling Required |
|---|---|---|
| **SVM** ⭐ | **82.52%** | Yes |
| Logistic Regression | 82.52% | Yes |
| Random Forest | 80.42% | No |
| Naive Bayes | 77.62% | No |
| Decision Tree | 76.22% | No |
| KNN | 79.02% | Yes |

### Breast Cancer Dataset (80/20 split, `random_state=42`)

| Algorithm | Accuracy | Scaling Required |
|---|---|---|
| **SVM** ⭐ | **98.25%** | Yes |
| Logistic Regression | 97.37% | Yes |
| KNN | 96.49% | Yes |
| Random Forest | 96.49% | No |
| Naive Bayes | 93.86% | No |
| Decision Tree | 93.86% | No |

> ⭐ **SVM wins both datasets.** Best models are auto-saved to `models/` as `.pkl` files.

---

## ⚙️ Algorithms Used

| Algorithm | Type | Notes |
|---|---|---|
| Logistic Regression | Linear | `max_iter=5000` |
| K-Nearest Neighbors | Instance-based | Default `k=5` |
| Support Vector Machine | Kernel-based | RBF kernel |
| Naive Bayes | Probabilistic | GaussianNB |
| Random Forest | Ensemble | 100 trees |
| Decision Tree | Tree-based | CART |

> **Note:** Logistic Regression, KNN, and SVM are trained on scaled features. Random Forest, Decision Tree, and Naive Bayes are trained on raw features.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models (via Jupyter)

```bash
jupyter lab
# Open notebooks/01_titanic_classification.ipynb
# Open notebooks/02_breast_cancer_classification.ipynb
```

### 3. Or train via Python scripts directly

```bash
python python_files/titanic_model.py
python python_files/breast_cancer_model.py
```

### 4. Launch the Streamlit app

```bash
cd app
streamlit run app.py
```

---

## 🖥️ Streamlit App

The app lets you make live predictions using the best saved model for each dataset.

**Titanic tab** — input passenger details (class, sex, age, fare, embarkation port) and predict survival.

**Breast Cancer tab** — input 30 cell nucleus measurements and predict malignant vs benign.

Best model info is shown in the sidebar with accuracy scores.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `numpy` / `pandas` | Data manipulation |
| `scikit-learn` | Model training, preprocessing, evaluation |
| `matplotlib` / `seaborn` | Visualization & dataset loading |
| `joblib` | Model serialization |
| `streamlit` | Interactive web app |
| `jupyterlab` | Notebook environment |

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

Install all with:

```bash
pip install -r requirements.txt
```

---

*Made with ❤️ using Streamlit + Scikit-learn*
