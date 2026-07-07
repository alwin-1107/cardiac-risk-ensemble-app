# Heart Disease Prediction — Ensemble Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://share.streamlit.io)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/alwin-1107/cardiac-risk-ensemble-app)

A predictive model for heart disease diagnosis built with ensemble machine
learning (Gradient Boosting, XGBoost, and a Stacking Classifier), deployed
as an interactive Streamlit web app.

**[Live Demo](https://cardiac-risk-ensemble.streamlit.app/)** · **[Full Report (PDF)](https://github.com/alwin-1107/cardiac-risk-ensemble-app/blob/main/reports/Ensemble_learning-heart-disease-prediction_Report.pdf)** · **[EDA Notebook](https://github.com/alwin-1107/cardiac-risk-ensemble-app/blob/main/notebooks/ensemble-learning-heat_disease_predictor-training.ipynb)**

> **Disclaimer:** This is an academic / portfolio project demonstrating
> ensemble learning techniques. It is **not** a medical device and should
> never be used for real clinical decision-making.

## Overview

The model learns from patient clinical data (age, cholesterol, chest pain
type, and other parameters) to predict the presence or absence of heart
disease.

- **Type:** Binary classification (`target`: 0 = No Disease, 1 = Disease)
- **Dataset:** [UCI Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) — 303 patient records, 13 clinical features
- **Metrics:** Accuracy, F1-Score, ROC-AUC

## Results

Three ensemble models were trained and compared using 5-fold cross-validation
on the training set, then evaluated on a held-out test set:

| Model              | Test Accuracy | Test F1-Score | Test ROC-AUC | CV-5 Mean Accuracy |
|---------------------|:---:|:---:|:---:|:---:|
| **Stacking (RF+SVM)** | **0.885** | **0.889** | **0.938** | 0.835 |
| Gradient Boosting    | 0.852 | 0.857 | 0.920 | 0.777 |
| XGBoost              | 0.836 | 0.844 | 0.890 | 0.806 |

The **Stacking Classifier** (Random Forest + SVM as base learners, Logistic
Regression as the meta-learner) was the best-performing and most robust
model, outperforming the individual boosting algorithms on every metric.
This is the model saved as `models/ensemble_model.pkl` and used by the app.

Detailed classification report for the Stacking model on the test set:

```
              precision    recall  f1-score   support

           0       0.87      0.90      0.88        29
           1       0.90      0.88      0.89        32

    accuracy                           0.89        61
   macro avg       0.88      0.89      0.89        61
weighted avg       0.89      0.89      0.89        61
```

## Methodology

**Preprocessing**
- Missing values (`?` in the raw data) imputed — median for numerical features, most-frequent for categorical features
- Categorical features one-hot encoded (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`)
- Numerical features standardized (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`)
- All steps combined into a single scikit-learn `Pipeline` + `ColumnTransformer` to prevent data leakage
- 80/20 train-test split

**Models compared**

| Model | Type | Description |
|---|---|---|
| Gradient Boosting Classifier | Boosting | Sequential ensemble, each tree corrects the previous one's errors |
| XGBoost Classifier | Boosting | Optimized, regularized gradient boosting |
| Stacking Classifier | Stacking | Combines base learner predictions via a meta-model |

**Stacking architecture**
- Base learners (Level 0): `RandomForestClassifier`, `SVC(probability=True)`
- Meta-learner (Level 1): `LogisticRegression`

## Repository structure

```
.
├── app.py                  # Streamlit web app
├── train.py                # Training script — reproduces the model from scratch
├── requirements.txt
├── data/
│   └── processed.cleveland.data   # UCI Cleveland dataset (checked in for reproducibility)
├── models/
│   └── ensemble_model.pkl         # Trained pipeline (preprocessing + Stacking model)
├── notebooks/
│   └── ensemble-learning-heat_disease_predictor-training.ipynb   # Original exploratory notebook (EDA, model comparison)
└── reports/
    └── Ensemble_learning-heart-disease-prediction_Report.pdf     # Full write-up: EDA, findings, comparison, conclusions
```

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/alwin-1107/cardiac-risk-ensemble-app.git
cd cardiac-risk-ensemble-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) retrain the model — takes well under a minute
python train.py

# 4. Launch the app
streamlit run app.py
```

The repo already ships with a trained `models/ensemble_model.pkl`, so step 3
is optional unless you want to regenerate it yourself.

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (including `models/ensemble_model.pkl` — it's
   ~800 KB, well within normal Git limits).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click **New app**.
3. Select this repository, branch `main`, and set the main file path to
   `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

## Future improvements

- **Hyperparameter tuning** — `GridSearchCV` or Bayesian optimization on the base and meta-models
- **Feature engineering** — interaction terms or polynomial features
- **Model interpretability** — SHAP or LIME to explain individual predictions, important for medical contexts
- **Testing** — unit tests for the preprocessing pipeline and app input handling

## Dataset citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart
Disease [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X
