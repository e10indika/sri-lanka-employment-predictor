import streamlit as st

# Configure multi-page app
st.set_page_config(page_title="Sri Lanka Employment Predictor", layout="wide")

# Sidebar navigation
st.sidebar.title("Sri Lanka Employment Predictor")
st.sidebar.markdown("Navigate through the app pages below.")

# Pages are auto-discovered from pages/ directory
# No need for manual selectbox; Streamlit handles it