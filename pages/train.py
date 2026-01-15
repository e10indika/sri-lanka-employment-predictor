"""
Train Model Page - Streamlit interface for training the employment prediction model.
"""
import streamlit as st
import pandas as pd
import sys
import os
from io import StringIO
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_preprocessing import DataPreprocessor
from modules.model_training import ModelTrainer
from modules.model_evaluation import ModelEvaluator
from config import MODEL_PARAMS, RAW_DATA_PATH, MODEL_CONFIGS


st.title("🎯 Train Employment Prediction Model")
st.markdown("Train a new model to predict employment status based on Sri Lankan labour force statistics.")

# Sidebar for training configuration
st.sidebar.header("Training Configuration")

# Model selection
st.sidebar.subheader("Model Selection")
train_all_models = st.sidebar.checkbox("Train All Models", value=False)

if train_all_models:
    st.sidebar.info("Will train all 6 models and compare results")
    model_type = None
else:
    model_type = st.sidebar.selectbox(
        "Choose Model",
        options=list(MODEL_CONFIGS.keys()),
        format_func=lambda x: MODEL_CONFIGS[x]['name'],
        index=0
    )
    st.sidebar.markdown(f"**Selected:** {MODEL_CONFIGS[model_type]['name']}")

# Training options
perform_cv = st.sidebar.checkbox("Perform Cross-Validation", value=True)
cv_folds = st.sidebar.slider("CV Folds", min_value=3, max_value=10, value=5)

perform_tuning = st.sidebar.checkbox("Hyperparameter Tuning", value=False)
if perform_tuning:
    st.sidebar.warning("⚠️ Tuning may take several minutes!")

# File upload
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload Employment Data CSV file",
    type=['csv'],
    help="Upload a CSV file with employment data (labour_force_stats_sri_lanka.csv format)."
)

use_default = st.checkbox("Use default dataset path", value=True)

# Display current configuration
with st.expander("📋 Current Model Configuration"):
    if train_all_models:
        st.markdown("**Training Mode:** All Models")
        for model_key, model_config in MODEL_CONFIGS.items():
            st.markdown(f"**{model_config['name']}**")
            st.json(model_config['params'])
    else:
        st.markdown(f"**Model:** {MODEL_CONFIGS[model_type]['name']}")
        st.json(MODEL_CONFIGS[model_type]['params'])

# Training button
if st.button("🚀 Start Training", type="primary", disabled=not (uploaded_file or use_default)):
    
    # Create placeholders for progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data Preprocessing
        status_text.text("📊 Step 1/3: Preprocessing data...")
        progress_bar.progress(10)
        
        preprocessor = DataPreprocessor()
        
        # Handle file upload
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = "temp_uploaded_data.csv"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            data_path = temp_path
        else:
            data_path = RAW_DATA_PATH if use_default else None
        
        with st.spinner("Loading and preprocessing data..."):
            X_train, X_test, y_train, y_test, df = preprocessor.prepare_data_pipeline(data_path)
        
        progress_bar.progress(30)
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Training Samples", len(X_train))
        with col3:
            st.metric("Test Samples", len(X_test))
        
        st.success("✅ Data preprocessing complete!")
        
        # Step 2: Model Training
        if train_all_models:
            status_text.text("🤖 Step 2/3: Training all models...")
            progress_bar.progress(40)
            
            # Train all models
            all_results = []
            models_to_train = list(MODEL_CONFIGS.keys())
            
            for idx, m_type in enumerate(models_to_train):
                m_name = MODEL_CONFIGS[m_type]['name']
                st.markdown(f"### Training {m_name}...")
                
                try:
                    trainer = ModelTrainer(model_type=m_type)
                    
                    # Cross-validation
                    if perform_cv:
                        with st.spinner(f"Cross-validating {m_name}..."):
                            cv_scores = trainer.cross_validate(X_train, y_train, cv=cv_folds)
                        st.info(f"📊 {m_name} CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
                    
                    # Train model
                    with st.spinner(f"Training {m_name}..."):
                        model = trainer.train_model(X_train, y_train, verbose=False)
                    
                    # Evaluate
                    evaluator = ModelEvaluator(model, X_test, y_test, model_type=m_type)
                    feature_names = preprocessor.feature_columns if hasattr(preprocessor, 'feature_columns') else None
                    metrics = evaluator.evaluate_model(feature_names=feature_names, generate_plots=True)
                    
                    all_results.append({
                        'Model': m_name,
                        'CV Accuracy': f"{cv_scores.mean():.4f}" if perform_cv else 'N/A',
                        'Test Accuracy': f"{metrics['accuracy']:.4f}",
                        'F1-Score': f"{metrics['f1_weighted']:.4f}",
                        'model_object': model,
                        'trainer': trainer
                    })
                    
                    st.success(f"✅ {m_name} complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error training {m_name}: {str(e)}")
                
                progress_bar.progress(0.4 + (0.3 * (idx + 1) / len(models_to_train)))
            
            # Display comparison
            st.markdown("---")
            st.header("📊 All Models Comparison")
            import pandas as pd
            results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model_object', 'trainer']} for r in all_results])
            st.dataframe(results_df, use_container_width=True)
            
            # Find best model
            best_result = max(all_results, key=lambda x: float(x['Test Accuracy']))
            st.success(f"🏆 Best Model: **{best_result['Model']}** with test accuracy {best_result['Test Accuracy']}")
            
            # Ask which model to save
            st.markdown("### Save Model")
            model_to_save = st.selectbox(
                "Select model to save as production model:",
                options=[r['Model'] for r in all_results],
                index=[r['Model'] for r in all_results].index(best_result['Model'])
            )
            
            if st.button("💾 Save Selected Model"):
                selected_result = next(r for r in all_results if r['Model'] == model_to_save)
                selected_result['trainer'].save_model()
                st.success(f"✅ {model_to_save} saved as production model!")
                st.balloons()
            
            progress_bar.progress(1.0)
            status_text.text("✅ All models trained!")
            
        else:
            # Single model training
            status_text.text(f"🤖 Step 2/3: Training {MODEL_CONFIGS[model_type]['name']} model...")
            progress_bar.progress(40)
            
            trainer = ModelTrainer(model_type=model_type)
        
        # Cross-validation
        if perform_cv:
            with st.spinner(f"Performing {cv_folds}-fold cross-validation..."):
                cv_scores = trainer.cross_validate(X_train, y_train, cv=cv_folds)
            
            st.info(f"📊 {trainer.model_name} CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
            progress_bar.progress(50)
        
        # Hyperparameter tuning or regular training
        if perform_tuning:
            with st.spinner("Tuning hyperparameters (this may take a while)..."):
                # Define a smaller grid for UI responsiveness
                if model_type == 'xgboost':
                    param_grid = {
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'n_estimators': [100, 200]
                    }
                elif model_type in ['random_forest', 'gradient_boosting']:
                    param_grid = {
                        'max_depth': [5, 10, 15],
                        'n_estimators': [100, 200]
                    }
                elif model_type == 'decision_tree':
                    param_grid = {
                        'max_depth': [5, 10, 15, 20]
                    }
                else:
                    param_grid = {}
                
                if param_grid:
                    best_params, model = trainer.hyperparameter_tuning(
                        X_train, y_train, param_grid=param_grid, cv=3
                    )
                    st.success("✅ Hyperparameter tuning complete!")
                    with st.expander("🎯 Best Parameters Found"):
                        st.json(best_params)
                else:
                    model = trainer.train_model(X_train, y_train, verbose=False)
                    st.info("ℹ️ No hyperparameter tuning available for this model type.")
            progress_bar.progress(70)
        else:
            with st.spinner(f"Training {trainer.model_name} with default parameters..."):
                model = trainer.train_model(X_train, y_train, verbose=False)
            
            st.success(f"✅ {trainer.model_name} training complete!")
            progress_bar.progress(70)
        
        # Save model
        trainer.save_model()
        st.success("✅ Model saved successfully!")
        
        # Step 3: Model Evaluation
        status_text.text("📈 Step 3/3: Evaluating model...")
        progress_bar.progress(75)
        
        with st.spinner("Evaluating model performance..."):
            evaluator = ModelEvaluator(model, X_test, y_test, model_type=model_type)
            metrics = evaluator.calculate_metrics()
        
        progress_bar.progress(85)
        
        # Display metrics
        st.header("📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision_weighted']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall_weighted']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_weighted']:.4f}")
        
        # Generate visualizations
        with st.spinner("Generating visualizations..."):
            # Get feature names from preprocessor
            feature_names = preprocessor.feature_columns if hasattr(preprocessor, 'feature_columns') else None
            evaluator.generate_all_visualizations(feature_names=feature_names)
        
        progress_bar.progress(100)
        status_text.text("✅ Training complete!")
        
        st.success("🎉 Training pipeline completed successfully!")
        
        # Display visualizations
        st.header("📈 Evaluation Visualizations")
        
        # Get model-specific visualization paths
        from config import get_model_viz_paths
        viz_paths = get_model_viz_paths(model_type)
        
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Feature Importance", "SHAP Summary"])
        
        with tab1:
            if os.path.exists(viz_paths['confusion_matrix']):
                st.image(viz_paths['confusion_matrix'], caption=f"Confusion Matrix - {trainer.model_name}", use_column_width=True)
        
        with tab2:
            if os.path.exists(viz_paths['feature_importance']):
                st.image(viz_paths['feature_importance'], caption=f"Feature Importance - {trainer.model_name}", use_column_width=True)
        
        with tab3:
            if os.path.exists(viz_paths['shap_summary']):
                st.image(viz_paths['shap_summary'], caption=f"SHAP Summary - {trainer.model_name}", use_column_width=True)
        
        # Detailed metrics
        with st.expander("📋 Detailed Classification Report"):
            # Get predictions for classification report
            y_pred = evaluator.predict()
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        
        # Clean up temporary file
        if uploaded_file and os.path.exists(temp_path):
            os.remove(temp_path)
        
        st.balloons()
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Please make sure the data file exists or upload a CSV file.")
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        import traceback
        with st.expander("View detailed error"):
            st.code(traceback.format_exc())

# Information section
st.markdown("---")
st.header("ℹ️ Information")

with st.expander("About the Training Process"):
    st.markdown("""
    ### Training Pipeline Steps:
    
    1. **Data Preprocessing**
       - Load and clean the dataset
       - Handle missing values and duplicates
       - Split into training and test sets
       - Scale features using StandardScaler
    
    2. **Model Training**
       - Train XGBoost classifier
       - Optional: Perform cross-validation
       - Optional: Hyperparameter tuning with GridSearchCV
    
    3. **Model Evaluation**
       - Calculate performance metrics
       - Generate confusion matrix
       - Visualize feature importance
       - Create SHAP summary for explainability
    
    ### Model Artifacts:
    - `model.pkl`: Trained model
    - `scaler.pkl`: Feature scaler
    - `confusion_matrix.png`: Confusion matrix visualization
    - `feature_importance.png`: Feature importance plot
    - `shap_summary.png`: SHAP explainability plot
    """)

with st.expander("Expected Data Format"):
    st.markdown("""
    ### CSV File Requirements:
    
    The CSV file should contain columns such as:
    - **SECTOR, DISTRICT, PSU, SERNO**: Geographic/survey identifiers
    - **SEX, AGE, MARITAL**: Demographics
    - **EDU, DEGREE, CUEDU**: Education level
    - **SIN, ENG, TAMIL**: Language proficiency (0/1)
    - **Eye Disability, Hearing Disability, Walking Disability, etc.**: Disability indicators (1-4 scale)
    - **Vocational Trained**: Vocational training status
    - **Employment, Employment_2**: Employment status indicators
    - **Unemployment Reason**: Reason for unemployment (if applicable)
    - **Certified On Employment**: Employment certification status
    
    The file should use comma (`,`) as delimiter.
    """)
