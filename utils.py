import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Import from config
from config import (
    MODEL_PATH, SCALER_PATH, CM_PATH, FI_PATH, 
    SHAP_PATH, SAMPLE_PATH
)

@st.cache_resource
def load_model_and_scaler(model_path=None, scaler_path=None):
    """Load the pickled model and scaler."""
    try:
        if model_path is None:
            model_path = MODEL_PATH
        if scaler_path is None:
            scaler_path = SCALER_PATH
            
        model_data = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Handle both old and new model formats
        if isinstance(model_data, dict):
            model = model_data['model']
        else:
            model = model_data
        
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None

def make_prediction(model, scaler, inputs):
    """Perform prediction and return SHAP explanation figure."""
    try:
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Local SHAP explanation
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(input_scaled)
            
            # Generate force plot (SHAP v0.20+)
            fig, ax = plt.subplots(figsize=(20, 3))
            
            # For binary classification, shap_values is an Explanation object
            # Pass it directly to force plot
            shap.plots.force(shap_values[0], matplotlib=True, show=False)
            fig = plt.gcf()  # Get the current figure created by SHAP
            plt.tight_layout()
        except Exception as shap_error:
            # If SHAP fails, still return prediction but without explanation
            st.warning(f"Could not generate SHAP explanation: {str(shap_error)}")
            fig = None
        
        return prediction, prediction_proba, fig
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def display_plot(path, caption):
    """Display an image plot if it exists."""
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.warning(f"{caption} plot not found.")

def load_sample_dataset():
    """Load and return sample dataset if exists."""
    if os.path.exists(SAMPLE_PATH):
        return pd.read_csv(SAMPLE_PATH)
    else:
        st.warning("Sample dataset not found.")
        return None