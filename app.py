"""
app.py
------
Streamlit front-end for the heart disease prediction ensemble model.

Loads models/ensemble_model.pkl (a scikit-learn Pipeline containing both
the preprocessing ColumnTransformer and the trained Stacking classifier)
and lets a user enter patient data through sliders / selectors to get a
prediction.

Run locally:
    streamlit run app.py
"""

import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ensemble_model.pkl")

# Project links -- update these if you rename/move the repo
GITHUB_REPO_URL = "https://github.com/alwin-1107/cardiac-risk-ensemble-app"
REPORT_URL = f"{GITHUB_REPO_URL}/blob/main/reports/Ensemble_learning-heart-disease-prediction_Report.pdf"
NOTEBOOK_URL = f"{GITHUB_REPO_URL}/blob/main/notebooks/ensemble-learning-heat_disease_predictor-training.ipynb"
DATASET_URL = "https://archive.ics.uci.edu/dataset/45/heart+disease"

# UI label -> encoded value, matching the UCI Cleveland dataset's original coding
SEX_OPTIONS = {"Female": 0, "Male": 1}
CP_OPTIONS = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-anginal Pain": 3,
    "Asymptomatic": 4,
}
FBS_OPTIONS = {"No (<= 120 mg/dl)": 0, "Yes (> 120 mg/dl)": 1}
RESTECG_OPTIONS = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2,
}
EXANG_OPTIONS = {"No": 0, "Yes": 1}
SLOPE_OPTIONS = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL_OPTIONS = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def render_sidebar():
    with st.sidebar:
        st.header("About This Project")
        st.markdown(
            "A Stacking Classifier (Random Forest + SVM, with a Logistic "
            "Regression meta-learner) trained on the UCI Cleveland Heart "
            "Disease dataset to predict presence of heart disease from "
            "13 clinical features."
        )

        st.subheader("Test Set Performance")
        # NOTE: these are stacked vertically (not st.columns(3)) on purpose.
        # The sidebar is narrow, and splitting into 3 side-by-side columns
        # left too little width per metric, causing st.metric to truncate
        # the values with an ellipsis (e.g. "88...", "0....").
        st.metric("Accuracy", "88.5%")
        st.metric("F1-Score", "0.889")
        st.metric("ROC-AUC", "0.938")

        st.subheader("Links")
        st.markdown(
            f"[\U0001F4C2 GitHub Repository]({GITHUB_REPO_URL})\n\n"
            f"[\U0001F4C4 Full Classification Report (PDF)]({REPORT_URL})\n\n"
            f"[\U0001F4D3 Original EDA Notebook]({NOTEBOOK_URL})\n\n"
            f"[\U0001F9EC Dataset (UCI Cleveland)]({DATASET_URL})"
        )

        st.caption("Built by Alwin - academic/portfolio project, not a diagnostic tool.")


def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="\U0001FAC0", layout="wide")

    render_sidebar()

    st.title("\U0001FAC0 Heart Disease Prediction")
    st.caption(
        "Ensemble machine learning model (Random Forest + SVM, stacked with a "
        "Logistic Regression meta-learner) trained on the UCI Cleveland Heart "
        "Disease dataset."
    )
    st.warning(
        "This is an academic / portfolio project for demonstrating ensemble "
        "learning. It is **not** a medical device and must not be used for "
        "real clinical decisions.",
        icon="\u26A0\uFE0F",
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at `{MODEL_PATH}`.\n\n"
            "Run `python train.py` first to train the model and generate it."
        )
        st.stop()

    model = load_model()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Clinical Data")
        age = st.slider("Age", 20, 90, 50)
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.2, 1.0, step=0.1)

    with col2:
        st.subheader("Categorical Data")
        sex_label = st.radio("Sex", list(SEX_OPTIONS.keys()), horizontal=True)
        cp_label = st.selectbox("Chest Pain Type", list(CP_OPTIONS.keys()))
        fbs_label = st.radio("Fasting Blood Sugar", list(FBS_OPTIONS.keys()), horizontal=True)
        restecg_label = st.selectbox("Resting ECG", list(RESTECG_OPTIONS.keys()))
        exang_label = st.radio("Exercise Induced Angina", list(EXANG_OPTIONS.keys()), horizontal=True)
        slope_label = st.selectbox("Slope of Peak Exercise ST Segment", list(SLOPE_OPTIONS.keys()))
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3])
        thal_label = st.selectbox("Thalassemia", list(THAL_OPTIONS.keys()))

    input_data = {
        "age": age,
        "sex": SEX_OPTIONS[sex_label],
        "cp": CP_OPTIONS[cp_label],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": FBS_OPTIONS[fbs_label],
        "restecg": RESTECG_OPTIONS[restecg_label],
        "thalach": thalach,
        "exang": EXANG_OPTIONS[exang_label],
        "oldpeak": oldpeak,
        "slope": SLOPE_OPTIONS[slope_label],
        "ca": ca,
        "thal": THAL_OPTIONS[thal_label],
    }
    input_df = pd.DataFrame([input_data])[FEATURE_ORDER]

    st.divider()

    if st.button("Predict", type="primary"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"The model predicts this patient **DOES** have heart disease. "
                      f"(Probability: {proba[1]:.2%})")
        else:
            st.success(f"The model predicts this patient **does NOT** have heart disease. "
                        f"(Probability of disease: {proba[1]:.2%})")

        st.caption("Model Input Data")
        st.dataframe(input_df, hide_index=True)

    st.divider()
    st.markdown(
        f"[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)]({GITHUB_REPO_URL}) "
        f"[![Report](https://img.shields.io/badge/Report-PDF-red)]({REPORT_URL}) "
        "[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Cleveland-blue)]"
        f"({DATASET_URL})"
    )


if __name__ == "__main__":
    main()
