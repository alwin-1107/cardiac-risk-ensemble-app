"""
train.py
--------
Trains and compares three ensemble classifiers (Gradient Boosting, XGBoost,
and a Stacking Classifier) on the UCI Cleveland Heart Disease dataset, then
saves the best-performing pipeline to models/ensemble_model.pkl.

This is a script version of notebooks/ensemble-learning-heat_disease_predictor-training.ipynb, with two practical
changes for reliable, repeatable runs outside of Colab:

1. Loads the dataset from the local data/processed.cleveland.data file that
   ships with this repo, instead of depending on a live UCI URL at run time.
   Falls back to the original UCI URL automatically if the local file is
   ever missing.
2. Drops the deprecated `use_label_encoder` argument from XGBClassifier.
   It's a no-op on current xgboost versions (>=1.6) and only produces a
   UserWarning; removing it just keeps the console output clean.

Everything else -- feature lists, preprocessing, model definitions,
hyperparameters, and evaluation -- matches the original notebook exactly,
so results are consistent with the project report.

Usage:
    python train.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_STATE = 42
LOCAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed.cleveland.data")
REMOTE_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "models", "ensemble_model.pkl")

COL_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def load_data() -> pd.DataFrame:
    """Load the Cleveland dataset from disk, falling back to the UCI URL."""
    if os.path.exists(LOCAL_DATA_PATH):
        print(f"Loading dataset from local file: {LOCAL_DATA_PATH}")
        df = pd.read_csv(LOCAL_DATA_PATH, header=None, names=COL_NAMES)
    else:
        print(f"Local data file not found, downloading from {REMOTE_DATA_URL}")
        df = pd.read_csv(REMOTE_DATA_URL, header=None, names=COL_NAMES)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_models(preprocessor: ColumnTransformer) -> dict:
    gb_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])

    xgb_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")),
    ])

    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ("svc", SVC(probability=True, random_state=RANDOM_STATE)),
    ]
    meta_learner = LogisticRegression()
    stacking_classifier = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
    )
    stacking_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", stacking_classifier),
    ])

    return {
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model,
        "Stacking (RF+SVM)": stacking_model,
    }


def main() -> None:
    df = load_data()

    # Missing values are marked as '?' in the raw UCI file
    df.replace("?", np.nan, inplace=True)

    # Binarize target: 0 = no disease, 1..4 -> 1 = disease present
    df["target"] = df["target"].apply(lambda x: 1 if int(float(x)) > 0 else 0)

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    results = {}
    print("\n--- Training & evaluating models ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Test F1-Score": f1_score(y_test, y_pred),
            "Test ROC-AUC": roc_auc_score(y_test, y_proba),
            "CV-5 Mean Accuracy": cross_val_score(
                model, X_train, y_train, cv=5, scoring="accuracy"
            ).mean(),
        }
        print(f"  Test Accuracy: {results[name]['Test Accuracy']:.4f}  "
              f"F1: {results[name]['Test F1-Score']:.4f}  "
              f"ROC-AUC: {results[name]['Test ROC-AUC']:.4f}")

    results_df = pd.DataFrame(results).T.sort_values(by="Test F1-Score", ascending=False)
    print("\n--- Final Model Comparison ---")
    print(results_df)

    best_model_name = results_df.index[0]
    best_model = models[best_model_name]

    print(f"\n--- Detailed Report for Best Model ({best_model_name}) ---")
    print(classification_report(y_test, best_model.predict(X_test)))

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"\nBest model ({best_model_name}) saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
