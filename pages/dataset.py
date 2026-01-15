import streamlit as st
import pandas as pd
from utils import load_sample_dataset

st.title("View and Upload Dataset")

st.markdown("""
This page allows you to view the sample employment dataset and upload your own data for analysis.
""")

# Sample dataset
sample_data = load_sample_dataset()
if sample_data is not None:
    st.subheader("Sample Dataset (First 100 Rows)")
    st.dataframe(sample_data, use_container_width=True)
    
    # Dataset statistics
    with st.expander("📊 Dataset Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(sample_data))
        with col2:
            st.metric("Total Columns", len(sample_data.columns))
        with col3:
            if 'Employment_Status_Encoded' in sample_data.columns:
                employed_pct = (sample_data['Employment_Status_Encoded'] == 1).mean() * 100
                st.metric("Employment Rate", f"{employed_pct:.1f}%")
    
    csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Sample Dataset", 
        data=csv, 
        file_name='employment_sample.csv', 
        mime='text/csv'
    )
else:
    st.info("Sample dataset will be generated after training the model.")

# User upload
st.markdown("---")
st.subheader("Upload Your Own Dataset")
uploaded_file = st.file_uploader(
    "Upload CSV file for analysis", 
    type='csv',
    help="Upload a CSV file with employment data. Should match the expected column structure."
)

if uploaded_file:
    try:
        user_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded file with {len(user_data)} rows and {len(user_data.columns)} columns")
        
        st.subheader("Uploaded Dataset Preview")
        st.dataframe(user_data.head(100), use_container_width=True)
        
        with st.expander("📋 Column Information"):
            st.write("Columns in uploaded file:")
            st.write(user_data.columns.tolist())
            
    except Exception as e:
        st.error(f"Error loading file: {e}")