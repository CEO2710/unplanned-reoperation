import streamlit as st
import pandas as pd
import joblib
import shap

# Load model
model = joblib.load("model.joblib")

# Variable configuration
VARIABLE_CONFIG = {
    "Sex": {"min": 0, "max": 1, "desc": "Patient gender (0=Female, 1=Male)"},
    "ASA scores": {"min": 0, "max": 5, "desc": "ASA physical status classification"},
    "tumor location": {"min": 1, "max": 4, "desc": "Tumor location code (1-4)"},
    "Benign or malignant": {"min": 0, "max": 1, "desc": "Tumor nature (0=Benign, 1=Malignant)"},
    "Admitted to NICU": {"min": 0, "max": 1, "desc": "NICU admission status"},
    "Duration of surgery": {"min": 0, "max": 1, "desc": "Surgery duration category"},
    "diabetes": {"min": 0, "max": 1, "desc": "Diabetes mellitus status"},
    "CHF": {"min": 0, "max": 1, "desc": "Congestive heart failure"},
    "Functional dependencies": {"min": 0, "max": 1, "desc": "Functional dependencies"},
    "mFI-5": {"min": 0, "max": 5, "desc": "Modified Frailty Index"},
    "Type of tumor": {"min": 1, "max": 5, "desc": "Tumor type code (1-5)"}
}

# Interface
st.set_page_config(page_title="Surgical Risk System")
st.title("Unplanned Reoperation Risk Prediction")

# Input widgets
inputs = {}
cols = st.columns(2)
for i, (feat, cfg) in enumerate(VARIABLE_CONFIG.items()):
    with cols[i % 2]:
        inputs[feat] = st.number_input(
            label=f"{feat} ({cfg['desc']})",
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["min"],
            step=1
        )

# Prediction
if st.button("Predict"):
    try:
        df = pd.DataFrame([inputs])
        proba = model.predict_proba(df)[0][1]
        risk = "High Risk" if proba > 0.5 else "Low Risk"
        
        st.success(f"Prediction: {risk} (Probability: {proba:.1%})")
        
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        st.pyplot(shap.force_plot(
            explainer.expected_value,
            shap_values[0], 
            df,
            matplotlib=True
        ))
        
    except Exception as e:
        st.error(f"Error: {str(e)}")