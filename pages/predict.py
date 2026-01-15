import streamlit as st
from utils import load_model_and_scaler, make_prediction
import sys
import os
import json

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.title("Employment Prediction")
st.subheader("Enter Individual Information")

# Get available models
available_models = config.get_available_models()

if len(available_models) == 0:
    st.error("No trained models found. Please train a model first.")
    st.stop()

# Model selection sidebar
st.sidebar.header("Model Selection")

if len(available_models) > 1:
    # Show model selector
    model_options = {m['model_name']: m['path'] for m in available_models}
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys())
    )
    selected_model_path = model_options[selected_model_name]
    st.sidebar.success(f"Using: {selected_model_name}")
else:
    # Only one model available
    selected_model_path = available_models[0]['path']
    selected_model_name = available_models[0]['model_name']
    st.sidebar.info(f"Using: {selected_model_name}")

# Display current model info
st.info(f"🤖 Using **{selected_model_name}** model for predictions")

result = load_model_and_scaler(model_path=selected_model_path)
if result[0] is None or result[1] is None:
    st.error("Model or scaler not found. Please train the model first.")
    st.stop()

model, scaler = result

# Reload feature columns from file (in case they were updated after app started)
FEATURE_COLUMNS = config.load_feature_columns()

# Check if we have feature columns defined
if not FEATURE_COLUMNS or len(FEATURE_COLUMNS) == 0:
    st.warning("⚠️ Feature columns not yet defined. Please train the model first to generate the feature set.")
    st.stop()

# Load feature info for better input widgets
feature_info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feature_info.json')
feature_info = {}
if os.path.exists(feature_info_path):
    with open(feature_info_path, 'r') as f:
        feature_info = json.load(f)

# Note about features
st.info("ℹ️ Enter values based on the training data ranges. Select boxes show available options.")

# Create input fields dynamically
st.markdown("### Input Features")
inputs = {}

# Use columns for better layout
col1, col2 = st.columns(2)

for i, feat in enumerate(FEATURE_COLUMNS):
    with col1 if i % 2 == 0 else col2:
        if feat in feature_info:
            info = feature_info[feat]
            label = info.get('label', feat)
            
            if info['type'] == 'select' and 'options' in info:
                # Use selectbox with labeled options
                options = info['options']
                # Convert string keys to int if needed
                options = {int(k) if isinstance(k, str) else k: v for k, v in options.items()}
                selected_label = st.selectbox(
                    label,
                    options=list(options.keys()),
                    format_func=lambda x: f"{x}: {options[x]}",
                    key=feat
                )
                inputs[feat] = selected_label
            elif info['type'] == 'select':
                # Use selectbox with just the numeric values
                inputs[feat] = st.selectbox(
                    label,
                    options=info['unique_values'],
                    key=feat
                )
            else:
                # Use number input with appropriate range
                inputs[feat] = st.number_input(
                    label,
                    min_value=info['min'],
                    max_value=info['max'],
                    value=info['min'],
                    step=1 if feat != 'AGE' else 1,
                    key=feat
                )
        else:
            # Fallback for features without info
            inputs[feat] = st.number_input(f"{feat}", value=0.0, step=0.1, key=feat)

if st.button("Predict Employment Status", type="primary"):
    with st.spinner("Making prediction..."):
        prediction, proba, fig = make_prediction(model, scaler, inputs)
        
        if prediction is not None:
            # Display result
            status = "Employed" if prediction == 1 else "Unemployed"
            status_color = "green" if prediction == 1 else "red"
            
            st.markdown(f"### Prediction: <span style='color:{status_color}; font-weight:bold'>{status}</span>", unsafe_allow_html=True)
            
            # Display probabilities
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability: Unemployed", f"{proba[0]:.2%}")
            with col2:
                st.metric("Probability: Employed", f"{proba[1]:.2%}")
            
            # SHAP explanation (if available)
            if fig is not None:
                st.subheader("Prediction Explanation (SHAP Force Plot)")
                st.pyplot(fig)
                st.info("The SHAP plot shows which features pushed the prediction towards Employed (red) or Unemployed (blue).")
            else:
                st.info("SHAP explanation is not available for this model.")

st.markdown("---")
st.markdown("""
**Note**: Predictions are based on the trained XGBoost model. The model predicts employment status based on various demographic and personal characteristics.

**Class Labels:**
- 0 = Unemployed
- 1 = Employed
""")