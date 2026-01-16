"""
Retrain all existing models with metrics data
This script retrains all model types to include metrics, CV scores, and training time
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_preprocessing import DataPreprocessor
from modules.model_training import ModelTrainer
from modules.model_evaluation import ModelEvaluator
from config import RAW_DATA_PATH, MODEL_CONFIGS
import time

def retrain_all_models():
    """Retrain all models with proper metrics"""
    print("=" * 60)
    print("Retraining All Models with Metrics")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n[1/3] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data_pipeline(RAW_DATA_PATH)
    print(f"✓ Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"✓ Features: {len(feature_names)}")
    
    results = []
    
    print(f"\n[2/3] Training {len(MODEL_CONFIGS)} models...")
    for i, (model_type, config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"\n[{i}/{len(MODEL_CONFIGS)}] Training {config['name']}...")
        
        try:
            start_time = time.time()
            
            # Initialize trainer
            trainer = ModelTrainer(model_type=model_type)
            
            # Cross-validation
            print(f"  → Running 5-fold cross-validation...")
            cv_scores = trainer.cross_validate(X_train, y_train, cv=5)
            print(f"  → CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Train model
            print(f"  → Training model...")
            trainer.train_model(X_train, y_train, verbose=False)
            
            # Evaluate
            print(f"  → Evaluating model...")
            evaluator = ModelEvaluator(trainer.model, X_test, y_test, model_type=model_type)
            metrics = evaluator.evaluate_model(feature_names=feature_names, generate_plots=True)
            
            training_time = time.time() - start_time
            
            # Save model with metrics
            print(f"  → Saving model with metrics...")
            trainer.save_model(X_train=X_train)
            
            results.append({
                'model_type': model_type,
                'model_name': config['name'],
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'cv_mean': cv_scores.mean(),
                'training_time': training_time
            })
            
            print(f"  ✓ {config['name']} completed in {training_time:.2f}s")
            print(f"    Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error training {config['name']}: {e}")
            results.append({
                'model_type': model_type,
                'model_name': config['name'],
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("[3/3] Training Summary")
    print("=" * 60)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    if successful:
        print(f"\n✓ Successfully trained {len(successful)} models:")
        print(f"\n{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'CV Mean':<12} {'Time (s)':<10}")
        print("-" * 71)
        for r in sorted(successful, key=lambda x: x['f1_score'], reverse=True):
            print(f"{r['model_name']:<25} {r['accuracy']:<12.4f} {r['f1_score']:<12.4f} {r['cv_mean']:<12.4f} {r['training_time']:<10.2f}")
        
        best_model = max(successful, key=lambda x: x['f1_score'])
        print(f"\n🏆 Best model: {best_model['model_name']} (F1: {best_model['f1_score']:.4f})")
    
    if failed:
        print(f"\n✗ Failed to train {len(failed)} models:")
        for r in failed:
            print(f"  - {r['model_name']}: {r['error']}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return successful, failed

if __name__ == "__main__":
    retrain_all_models()
