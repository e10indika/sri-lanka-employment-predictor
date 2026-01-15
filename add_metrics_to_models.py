#!/usr/bin/env python3
"""
Utility script to add metrics to existing trained models.
This evaluates existing models and saves them with metrics for comparison.
"""
import sys
import os
import joblib
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_preprocessing import DataPreprocessor
from modules.model_evaluation import ModelEvaluator
from config import get_available_models

def add_metrics_to_model(model_path, X_test, y_test):
    """Load a model, evaluate it, and save with metrics."""
    print(f"\nProcessing: {model_path}")
    
    try:
        # Load existing model
        model_data = joblib.load(model_path)
        
        # Check if it's a dict or just the model
        if isinstance(model_data, dict):
            if 'metrics' in model_data:
                print(f"  ✓ Model already has metrics, skipping")
                return True
            
            model = model_data['model']
            model_type = model_data.get('model_type', 'unknown')
            model_name = model_data.get('model_name', 'Unknown Model')
        else:
            # Old format - just the model object
            model = model_data
            # Infer type from filename
            filename = Path(model_path).stem
            model_type = filename.replace('model_', '')
            model_name = model_type.replace('_', ' ').title()
            model_data = {
                'model': model,
                'model_type': model_type,
                'model_name': model_name
            }
        
        print(f"  Evaluating {model_name}...")
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='binary', zero_division=0))
        }
        
        training_time = time.time() - start_time
        
        # Update model data
        model_data['metrics'] = metrics
        model_data['training_time'] = training_time
        model_data['cv_scores'] = []  # No CV scores for existing models
        
        # Save updated model
        joblib.dump(model_data, model_path)
        
        print(f"  ✓ Metrics added successfully:")
        print(f"    - Accuracy: {metrics['accuracy']:.4f}")
        print(f"    - Precision: {metrics['precision']:.4f}")
        print(f"    - Recall: {metrics['recall']:.4f}")
        print(f"    - F1 Score: {metrics['f1_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to process all models."""
    print("=" * 70)
    print("Adding Metrics to Existing Models")
    print("=" * 70)
    
    # Load test data
    print("\nLoading data...")
    try:
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data_pipeline()
        print(f"✓ Loaded test data: {X_test.shape[0]} samples")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Get all available models
    models = get_available_models()
    
    if not models:
        print("\n✗ No models found in the models directory")
        return 1
    
    print(f"\nFound {len(models)} models to process")
    
    # Process each model
    success_count = 0
    failed_count = 0
    
    for model_info in models:
        if add_metrics_to_model(model_info['path'], X_test, y_test):
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Successfully processed: {success_count} models")
    if failed_count > 0:
        print(f"✗ Failed: {failed_count} models")
    
    print("\n✓ Done! You can now use the Compare Models feature.")
    print("  Visit: http://localhost:3000/compare-models")
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
