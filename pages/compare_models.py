"""
Model Comparison Page - Compare performance of different ML models
"""
import streamlit as st
import pandas as pd
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_preprocessing import DataPreprocessor
from modules.model_training import ModelTrainer
from modules.model_evaluation import ModelEvaluator
from config import MODEL_CONFIGS, RAW_DATA_PATH

st.title("⚖️ Model Comparison")
st.markdown("Compare the performance of different machine learning models on employment prediction.")

# Model selection
st.sidebar.header("Select Models to Compare")
selected_models = []
for model_key, model_config in MODEL_CONFIGS.items():
    if st.sidebar.checkbox(model_config['name'], value=(model_key in ['xgboost', 'random_forest', 'decision_tree'])):
        selected_models.append(model_key)

cv_folds = st.sidebar.slider("Cross-Validation Folds", 3, 10, 5)

# Start comparison button
if st.button("🚀 Start Comparison", type="primary", disabled=len(selected_models) == 0):
    if len(selected_models) == 0:
        st.warning("⚠️ Please select at least one model to compare.")
    else:
        st.info(f"Comparing {len(selected_models)} models: {', '.join([MODEL_CONFIGS[m]['name'] for m in selected_models])}")
        
        # Preprocessing
        with st.spinner("Loading and preprocessing data..."):
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test, df = preprocessor.prepare_data_pipeline(RAW_DATA_PATH)
        
        st.success(f"✅ Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
        
        # Results storage
        results = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train and evaluate each model
        for idx, model_type in enumerate(selected_models):
            model_name = MODEL_CONFIGS[model_type]['name']
            status_text.text(f"Training {model_name}... ({idx+1}/{len(selected_models)})")
            
            try:
                # Train model
                trainer = ModelTrainer(model_type=model_type)
                
                # Cross-validation
                cv_scores = trainer.cross_validate(X_train, y_train, cv=cv_folds)
                
                # Train on full training set
                model = trainer.train_model(X_train, y_train, verbose=False)
                
                # Evaluate on test set
                evaluator = ModelEvaluator(model, X_test, y_test, model_type=model_type)
                feature_names = preprocessor.feature_columns if hasattr(preprocessor, 'feature_columns') else None
                metrics = evaluator.evaluate_model(feature_names=feature_names, generate_plots=True)
                
                # Store results
                results.append({
                    'Model': model_name,
                    'CV Accuracy': f"{cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})",
                    'Test Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision (Weighted)': f"{metrics['precision_weighted']:.4f}",
                    'Recall (Weighted)': f"{metrics['recall_weighted']:.4f}",
                    'F1-Score (Weighted)': f"{metrics['f1_weighted']:.4f}",
                    'Training Time': 'N/A'
                })
                
                progress_bar.progress((idx + 1) / len(selected_models))
                
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                results.append({
                    'Model': model_name,
                    'CV Accuracy': 'Error',
                    'Test Accuracy': 'Error',
                    'Precision (Weighted)': 'Error',
                    'Recall (Weighted)': 'Error',
                    'F1-Score (Weighted)': 'Error',
                    'Training Time': 'Error'
                })
        
        status_text.text("✅ Comparison complete!")
        progress_bar.progress(1.0)
        
        # Display results
        st.header("📊 Comparison Results")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Highlight best model
        st.subheader("🏆 Best Performing Model")
        
        # Parse accuracy values for comparison
        try:
            accuracy_values = []
            for r in results:
                try:
                    acc_str = r['Test Accuracy']
                    if acc_str != 'Error':
                        accuracy_values.append((r['Model'], float(acc_str)))
                except:
                    pass
            
            if accuracy_values:
                best_model = max(accuracy_values, key=lambda x: x[1])
                st.success(f"**{best_model[0]}** achieved the highest test accuracy: **{best_model[1]:.4f}**")
                
                # Show ranking
                accuracy_values.sort(key=lambda x: x[1], reverse=True)
                st.markdown("### 📈 Model Ranking (by Test Accuracy)")
                for rank, (model, acc) in enumerate(accuracy_values, 1):
                    emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "▪️"
                    st.markdown(f"{emoji} **{rank}.** {model}: {acc:.4f}")
        except Exception as e:
            st.warning(f"Could not determine best model: {str(e)}")
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Results (CSV)",
            data=csv,
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        st.markdown("""
        **Model Selection Guide:**
        - **XGBoost/Gradient Boosting**: Usually highest accuracy, good for production
        - **Random Forest**: Robust, less prone to overfitting, good interpretability
        - **Decision Tree**: Fast, highly interpretable, but may overfit
        - **Logistic Regression**: Simple, fast, good baseline
        - **Naive Bayes**: Very fast, works well with smaller datasets
        
        Consider both **accuracy** and **training time** when selecting your production model.
        """)

else:
    st.info("👈 Select models from the sidebar and click 'Start Comparison' to begin.")
    
    # Display available models
    st.markdown("### 📋 Available Models")
    for model_key, model_config in MODEL_CONFIGS.items():
        with st.expander(f"{model_config['name']}"):
            st.markdown(f"**Type:** `{model_key}`")
            st.json(model_config['params'])
