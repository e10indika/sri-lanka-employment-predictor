"""
Model evaluation module for Sri Lanka Employment Predictor.
Handles model evaluation, visualization, and performance metrics.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import shap
import joblib
from lime import lime_tabular
from sklearn.inspection import PartialDependenceDisplay
from config import CM_PATH, FI_PATH, SHAP_PATH, get_model_viz_paths


class ModelEvaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self, model, X_test, y_test, model_type=None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_type: Type of model (for model-specific visualizations)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.metrics = {}
        self.model_type = model_type
        
        # Get model-specific visualization paths if model_type is provided
        if model_type:
            self.viz_paths = get_model_viz_paths(model_type)
            # Add LIME and PDP paths in root VIZ_DIR with model_type suffix
            from config import VIZ_DIR
            import os
            self.viz_paths['lime_explanation'] = os.path.join(VIZ_DIR, f'lime_explanation_{model_type}.png')
            self.viz_paths['pdp_plot'] = os.path.join(VIZ_DIR, f'partial_dependence_{model_type}.png')
        else:
            self.viz_paths = {
                'confusion_matrix': CM_PATH,
                'feature_importance': FI_PATH,
                'shap_summary': SHAP_PATH
            }
    
    def predict(self):
        """Make predictions on test set."""
        # Handle models with or without dict wrapper
        if hasattr(self.model, 'predict'):
            self.y_pred = self.model.predict(self.X_test).astype(int)
        else:
            # If model is wrapped in dict
            if isinstance(self.model, dict):
                model_obj = self.model['model']
                self.y_pred = model_obj.predict(self.X_test).astype(int)
            else:
                self.y_pred = self.model.predict(self.X_test).astype(int)
        
        return self.y_pred
    
    def calculate_metrics(self):
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self.y_pred is None:
            self.predict()
        
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision_weighted': precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(self.y_test, self.y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(self.y_test, self.y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        }
        
        return self.metrics
    
    def print_metrics(self):
        """Print all evaluation metrics."""
        if not self.metrics:
            self.calculate_metrics()
        
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        print(f"\nAccuracy:           {self.metrics['accuracy']:.4f}")
        print(f"\nWeighted Metrics:")
        print(f"  Precision:        {self.metrics['precision_weighted']:.4f}")
        print(f"  Recall:           {self.metrics['recall_weighted']:.4f}")
        print(f"  F1-Score:         {self.metrics['f1_weighted']:.4f}")
        
        print(f"\nMacro Metrics:")
        print(f"  Precision:        {self.metrics['precision_macro']:.4f}")
        print(f"  Recall:           {self.metrics['recall_macro']:.4f}")
        print(f"  F1-Score:         {self.metrics['f1_macro']:.4f}")
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(self.y_test, self.y_pred, zero_division=0))
        
        return self.metrics
    
    def plot_confusion_matrix(self, save_path=None, figsize=(10, 8)):
        """
        Plot and save confusion matrix.
        
        Args:
            save_path: Path to save plot (uses model-specific path if None)
            figsize: Figure size tuple
        """
        if self.y_pred is None:
            self.predict()
        
        if save_path is None:
            save_path = self.viz_paths['confusion_matrix']
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Get model name for title
        if isinstance(self.model, dict):
            model_name = self.model.get('model_name', type(self.model['model']).__name__)
        else:
            model_name = type(self.model).__name__
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=sorted(self.y_test.unique()),
                    yticklabels=sorted(self.y_test.unique()))
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_feature_importance(self, feature_names=None, save_path=None, 
                                 top_n=15, figsize=(10, 8)):
        """
        Plot and save feature importance.
        
        Args:
            feature_names: List of feature names
            save_path: Path to save plot (uses model-specific path if None)
            top_n: Number of top features to display
            figsize: Figure size tuple
        """
        if save_path is None:
            save_path = self.viz_paths['feature_importance']
        
        # Get the actual model object
        if isinstance(self.model, dict):
            model_obj = self.model['model']
        else:
            model_obj = self.model
        
        # Get feature importance based on model type
        try:
            if hasattr(model_obj, 'feature_importances_'):
                # Tree-based models (XGBoost, Random Forest, Decision Tree, Gradient Boosting)
                importance = model_obj.feature_importances_
                model_name = type(model_obj).__name__
            elif hasattr(model_obj, 'coef_'):
                # Linear models (Logistic Regression, etc.)
                importance = np.abs(model_obj.coef_[0])
                model_name = 'Logistic Regression'
            else:
                # Models without feature importance (e.g., Naive Bayes)
                print(f"Model type {type(model_obj).__name__} does not support feature importance plotting.")
                return None
        except Exception as e:
            print(f"Could not extract feature importance: {str(e)}")
            return None
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance))]
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Feature Importance ({model_name})', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to {save_path}")
        
        return importance_df
    
    def plot_shap_summary(self, feature_names=None, save_path=None, 
                          max_display=15, sample_size=None):
        """
        Generate and save SHAP summary plot.
        
        Args:
            feature_names: List of feature names
            save_path: Path to save plot (uses model-specific path if None)
            max_display: Maximum number of features to display
            sample_size: Number of samples to use for SHAP (None = all)
        """
        if save_path is None:
            save_path = self.viz_paths['shap_summary']
        
        print("Generating SHAP values (this may take a few moments)...")
        
        # Get the actual model object
        if isinstance(self.model, dict):
            model_obj = self.model['model']
        else:
            model_obj = self.model
        
        # Sample data if specified
        if sample_size and sample_size < len(self.X_test):
            indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test[indices]
        else:
            X_sample = self.X_test
        
        # Create explainer based on model type
        try:
            model_type = type(model_obj).__name__
            
            if model_type in ['LogisticRegression', 'GaussianNB']:
                # Use LinearExplainer for linear models or KernelExplainer as fallback
                try:
                    # For linear models, use LinearExplainer
                    explainer = shap.LinearExplainer(model_obj, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                except:
                    # Fallback to KernelExplainer (slower but works for all models)
                    print(f"Using KernelExplainer for {model_type} (this may take longer)...")
                    explainer = shap.KernelExplainer(model_obj.predict_proba, shap.sample(X_sample, 100))
                    shap_values = explainer.shap_values(X_sample)
            else:
                # Tree-based models use TreeExplainer or default Explainer
                explainer = shap.Explainer(model_obj)
                shap_values = explainer(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, shap.Explanation):
                # New SHAP format
                shap_values_array = shap_values.values
            elif isinstance(shap_values, list):
                # For binary classification, use positive class
                shap_values_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_array = shap_values
            
            # Create and save plot
            plt.figure(figsize=(12, 8))
            
            if isinstance(shap_values, shap.Explanation):
                shap.summary_plot(shap_values, X_sample, 
                                feature_names=feature_names,
                                max_display=max_display, 
                                show=False)
            else:
                shap.summary_plot(shap_values_array, X_sample, 
                                feature_names=feature_names,
                                max_display=max_display, 
                                show=False)
            
            plt.title(f'SHAP Summary Plot - {model_type}', 
                    fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP summary plot saved to {save_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate SHAP plot for {type(model_obj).__name__}: {str(e)}")
            print("Skipping SHAP visualization for this model.")
    
    def generate_lime_explanation(self, X_train, feature_names=None, save_path=None, 
                                   num_samples=5, num_features=10):
        """
        Generate LIME explanations for sample predictions.
        
        Args:
            X_train: Training data (used as background for LIME)
            feature_names: List of feature names
            save_path: Path to save plot (uses model-specific path if None)
            num_samples: Number of test samples to explain
            num_features: Number of top features to show in explanation
        """
        if save_path is None:
            save_path = self.viz_paths.get('lime_explanation', 'visualizations/lime_explanation.png')
        
        print("Generating LIME explanations...")
        
        # Get the actual model object
        if isinstance(self.model, dict):
            model_obj = self.model['model']
        else:
            model_obj = self.model
        
        try:
            # Convert to numpy arrays if needed
            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_np = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test
            
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X_train_np,
                feature_names=feature_names if feature_names else [f'Feature_{i}' for i in range(X_train_np.shape[1])],
                class_names=['Unemployed', 'Employed'],
                mode='classification',
                random_state=42
            )
            
            # Select random samples to explain
            sample_indices = np.random.choice(len(X_test_np), 
                                            min(num_samples, len(X_test_np)), 
                                            replace=False)
            
            # Create subplots for multiple explanations
            fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
            if num_samples == 1:
                axes = [axes]
            
            for idx, sample_idx in enumerate(sample_indices):
                # Get explanation for this instance
                exp = explainer.explain_instance(
                    X_test_np[sample_idx],
                    model_obj.predict_proba,
                    num_features=num_features
                )
                
                # Get prediction
                pred = model_obj.predict([X_test_np[sample_idx]])[0]
                pred_proba = model_obj.predict_proba([X_test_np[sample_idx]])[0]
                actual = self.y_test.iloc[sample_idx] if hasattr(self.y_test, 'iloc') else self.y_test[sample_idx]
                
                # Plot explanation
                exp_list = exp.as_list()
                features = [f[0] for f in exp_list]
                weights = [f[1] for f in exp_list]
                
                colors = ['green' if w > 0 else 'red' for w in weights]
                axes[idx].barh(range(len(features)), weights, color=colors, alpha=0.6)
                axes[idx].set_yticks(range(len(features)))
                axes[idx].set_yticklabels(features, fontsize=9)
                axes[idx].set_xlabel('Feature Weight', fontsize=10)
                axes[idx].set_title(
                    f'Sample {sample_idx}: Predicted={pred} (prob={pred_proba[pred]:.3f}), Actual={int(actual)}',
                    fontsize=11, fontweight='bold'
                )
                axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                axes[idx].grid(axis='x', alpha=0.3)
            
            plt.suptitle('LIME Explanations - Individual Predictions', 
                        fontsize=16, fontweight='bold', y=1.0)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"LIME explanations saved to {save_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate LIME explanations: {str(e)}")
            print("Skipping LIME visualization.")
    
    def plot_partial_dependence(self, X_train, feature_names=None, save_path=None, 
                                features_to_plot=None, num_features=4):
        """
        Generate Partial Dependence Plots (PDP) for top features.
        
        Args:
            X_train: Training data (used for PDP computation)
            feature_names: List of feature names
            save_path: Path to save plot (uses model-specific path if None)
            features_to_plot: Specific feature indices to plot (None = auto-select top features)
            num_features: Number of features to plot if features_to_plot is None
        """
        if save_path is None:
            save_path = self.viz_paths.get('pdp_plot', 'visualizations/partial_dependence.png')
        
        print("Generating Partial Dependence Plots...")
        
        # Get the actual model object
        if isinstance(self.model, dict):
            model_obj = self.model['model']
        else:
            model_obj = self.model
        
        try:
            # Convert to numpy arrays if needed
            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            
            # Determine which features to plot
            if features_to_plot is None:
                # Select top features based on feature importance
                if hasattr(model_obj, 'feature_importances_'):
                    importance = model_obj.feature_importances_
                    top_indices = np.argsort(importance)[-num_features:][::-1]
                    features_to_plot = top_indices.tolist()
                elif hasattr(model_obj, 'coef_'):
                    importance = np.abs(model_obj.coef_[0])
                    top_indices = np.argsort(importance)[-num_features:][::-1]
                    features_to_plot = top_indices.tolist()
                else:
                    # Default to first num_features
                    features_to_plot = list(range(min(num_features, X_train_np.shape[1])))
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X_train_np.shape[1])]
            
            # Generate PDP
            fig, ax = plt.subplots(figsize=(14, 10))
            
            display = PartialDependenceDisplay.from_estimator(
                model_obj,
                X_train_np,
                features=features_to_plot,
                feature_names=feature_names,
                n_cols=2,
                grid_resolution=50,
                ax=ax
            )
            
            plt.suptitle('Partial Dependence Plots - Feature Effects on Prediction', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Partial Dependence Plots saved to {save_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate Partial Dependence Plots: {str(e)}")
            print("Skipping PDP visualization.")
    
    def generate_all_visualizations(self, feature_names=None, X_train=None):
        """
        Generate all evaluation visualizations including LIME and PDP.
        
        Args:
            feature_names: List of feature names
            X_train: Training data (needed for LIME and PDP)
        """
        print("\nGenerating evaluation visualizations...")
        
        # Confusion matrix
        self.plot_confusion_matrix()
        
        # Feature importance
        self.plot_feature_importance(feature_names)
        
        # SHAP summary
        self.plot_shap_summary(feature_names, sample_size=500)
        
        # LIME explanations (if training data provided)
        if X_train is not None:
            self.generate_lime_explanation(X_train, feature_names, num_samples=3)
        
        # Partial Dependence Plots (if training data provided)
        if X_train is not None:
            self.plot_partial_dependence(X_train, feature_names, num_features=4)
        
        print("\nAll visualizations generated successfully!")
    
    def evaluate_model(self, feature_names=None, generate_plots=True, X_train=None):
        """
        Complete evaluation pipeline.
        
        Args:
            feature_names: List of feature names
            generate_plots: Whether to generate visualizations
            X_train: Training data (needed for LIME and PDP)
        
        Returns:
            Dictionary of metrics
        """
        print("\n" + "="*60)
        print("STARTING MODEL EVALUATION")
        print("="*60)
        
        # Calculate and print metrics
        self.calculate_metrics()
        self.print_metrics()
        
        # Generate visualizations
        if generate_plots:
            self.generate_all_visualizations(feature_names, X_train)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        return self.metrics


if __name__ == "__main__":
    # Test the evaluator
    from modules.data_preprocessing import DataPreprocessor
    from modules.model_training import ModelTrainer
    from config import FEATURE_COLUMNS
    
    print("Testing Model Evaluator...")
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data_pipeline()
    
    # Train model
    trainer = ModelTrainer()
    model = trainer.train_model(X_train, y_train)
    
    # Evaluate model
    evaluator = ModelEvaluator(model, X_test, y_test)
    metrics = evaluator.evaluate_model(feature_names=FEATURE_COLUMNS)
    
    print("\nEvaluation test complete!")
