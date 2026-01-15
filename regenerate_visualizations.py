"""
Regenerate model-specific visualizations for all existing trained models.
This script re-evaluates all trained models and generates their visualizations
with model-specific filenames.
"""
import sys
import os

from modules.data_preprocessing import DataPreprocessor
from modules.model_evaluation import ModelEvaluator
from config import get_available_models, SCALER_PATH
import joblib

def regenerate_all_visualizations():
    """Regenerate visualizations for all trained models."""
    
    print("=" * 60)
    print("REGENERATING MODEL-SPECIFIC VISUALIZATIONS")
    print("=" * 60)
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        print("❌ No trained models found.")
        return
    
    print(f"\nFound {len(available_models)} trained models:")
    for model in available_models:
        print(f"  - {model['model_name']} ({model['filename']})")
    
    # Load data
    print("\n📊 Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data_pipeline()
    feature_names = preprocessor.feature_columns
    
    print(f"✅ Data loaded: {len(X_test)} test samples")
    
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Scaler loaded")
    
    # Regenerate visualizations for each model
    for idx, model_info in enumerate(available_models, 1):
        model_name = model_info['model_name']
        model_type = model_info['model_type']
        model_path = model_info['path']
        
        print(f"\n[{idx}/{len(available_models)}] Processing {model_name}...")
        
        try:
            # Load model
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                model = model_data['model']
            else:
                model = model_data
            
            print(f"  ✓ Model loaded")
            
            # Create evaluator with model_type for model-specific paths
            evaluator = ModelEvaluator(model, X_test, y_test, model_type=model_type)
            
            # Generate visualizations
            print(f"  ⏳ Generating visualizations...")
            evaluator.generate_all_visualizations(feature_names=feature_names)
            
            print(f"  ✅ {model_name} visualizations complete!")
            
        except Exception as e:
            print(f"  ❌ Error processing {model_name}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ VISUALIZATION REGENERATION COMPLETE")
    print("=" * 60)
    print("\nAll models now have separate visualization files:")
    print("  - confusion_matrix_{model_type}.png")
    print("  - feature_importance_{model_type}.png")
    print("  - shap_summary_{model_type}.png")
    print("\nView them in the Dashboard page with model selection!")


if __name__ == "__main__":
    regenerate_all_visualizations()
