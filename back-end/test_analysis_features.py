"""
Test script to demonstrate LIME, Feature Importance, and PDP analysis
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_preprocessing import DataPreprocessor
from modules.model_training import ModelTrainer
from modules.model_evaluation import ModelEvaluator
from config import MODEL_DIR

def main():
    print("="*70)
    print("TESTING ENHANCED MODEL ANALYSIS FEATURES")
    print("="*70)
    
    # 1. Load and prepare data
    print("\n[1/5] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, df = preprocessor.prepare_data_pipeline()
    feature_names = preprocessor.feature_columns
    print(f"✓ Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"✓ Features: {len(feature_names)}")
    
    # 2. Train model (or load existing)
    print("\n[2/5] Training LightGBM model...")
    trainer = ModelTrainer(model_type='lightgbm')
    
    # Check if model already exists
    model_path = os.path.join(MODEL_DIR, 'lightgbm', 'model.pkl')
    if os.path.exists(model_path):
        print("✓ Loading existing model...")
        model = trainer.load_model(model_path)
    else:
        print("✓ Training new model...")
        model = trainer.train_model(X_train, y_train, verbose=False)
        # Perform cross-validation
        cv_scores = trainer.cross_validate(X_train, y_train, cv=5)
        print(f"✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        # Save model
        trainer.save_model(X_train=X_train)
        print("✓ Model saved with SHAP explainer")
    
    # 3. Create evaluator
    print("\n[3/5] Initializing model evaluator...")
    evaluator = ModelEvaluator(model, X_test, y_test, model_type='lightgbm')
    
    # 4. Generate all visualizations
    print("\n[4/5] Generating visualizations...")
    print("\n  a) Confusion Matrix...")
    evaluator.plot_confusion_matrix()
    
    print("\n  b) Feature Importance...")
    importance_df = evaluator.plot_feature_importance(feature_names)
    if importance_df is not None:
        print("\n  Top 5 Features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
    
    print("\n  c) SHAP Summary Plot...")
    evaluator.plot_shap_summary(feature_names, sample_size=500)
    
    print("\n  d) LIME Explanations...")
    evaluator.generate_lime_explanation(
        X_train, 
        feature_names, 
        num_samples=3,
        num_features=10
    )
    
    print("\n  e) Partial Dependence Plots...")
    evaluator.plot_partial_dependence(
        X_train, 
        feature_names,
        num_features=4
    )
    
    # 5. Calculate and display metrics
    print("\n[5/5] Calculating performance metrics...")
    metrics = evaluator.calculate_metrics()
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}")
    print(f"F1 Score (Weighted):  {metrics['f1_weighted']:.4f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - Confusion Matrix: visualizations/lightgbm/confusion_matrix_lightgbm.png")
    print("  - Feature Importance: visualizations/lightgbm/feature_importance_lightgbm.png")
    print("  - SHAP Summary: visualizations/lightgbm/shap_summary_lightgbm.png")
    print("  - LIME Explanations: visualizations/lightgbm/lime_explanation.png")
    print("  - Partial Dependence: visualizations/lightgbm/partial_dependence.png")
    
    print("\n✓ All analysis features tested successfully!")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
