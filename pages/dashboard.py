import streamlit as st
from config import get_available_models, get_model_viz_paths
from utils import display_plot
import os

st.title("Model Dashboard")
st.subheader("Employment Prediction Model Performance")

st.markdown("""
View the trained models' performance metrics and explainability visualizations.
The models predict employment status (Employed vs Unemployed) based on demographic 
and personal characteristics from Sri Lankan labour force statistics.
""")

st.markdown("---")

# Get available models
available_models = get_available_models()

if not available_models:
    st.warning("⚠️ No trained models found. Please train a model first.")
    st.stop()

# Model selector
st.sidebar.header("Select Model")
model_options = {model['model_name']: model for model in available_models}
selected_model_name = st.sidebar.selectbox(
    "Choose a model to view",
    options=list(model_options.keys())
)

selected_model = model_options[selected_model_name]
model_type = selected_model['model_type']

# Display model info
st.info(f"📊 **Viewing:** {selected_model_name} ({selected_model['filename']})")

# Get model-specific visualization paths
viz_paths = get_model_viz_paths(model_type)

# Display visualizations
st.subheader("Model Evaluation Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Confusion Matrix")
    if os.path.exists(viz_paths['confusion_matrix']):
        display_plot(viz_paths['confusion_matrix'], f'Confusion Matrix - {selected_model_name}')
    else:
        st.warning(f"Confusion matrix not found for {selected_model_name}. Train and evaluate this model first.")

with col2:
    st.markdown("### Feature Importance")
    if os.path.exists(viz_paths['feature_importance']):
        display_plot(viz_paths['feature_importance'], f'Feature Importance - {selected_model_name}')
    else:
        st.warning(f"Feature importance plot not found for {selected_model_name}. Train and evaluate this model first.")

st.markdown("### SHAP Summary")
if os.path.exists(viz_paths['shap_summary']):
    display_plot(viz_paths['shap_summary'], f'SHAP Summary - {selected_model_name}')
else:
    st.warning(f"SHAP summary plot not found for {selected_model_name}. Train and evaluate this model first.")

st.markdown("---")
st.markdown("""
**Model Insights:**
- The confusion matrix shows how well the model distinguishes between employed and unemployed individuals
- Feature importance reveals which factors most strongly influence employment predictions
- SHAP values provide interpretable explanations for how each feature contributes to predictions
""")

# Show all available models
with st.expander("📋 All Available Models"):
    for model in available_models:
        has_viz = all(os.path.exists(path) for path in get_model_viz_paths(model['model_type']).values())
        viz_status = "✅ Visualizations available" if has_viz else "⚠️ Needs evaluation"
        st.markdown(f"- **{model['model_name']}** ({model['filename']}) - {viz_status}")