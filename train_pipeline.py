"""
Complete training pipeline for Sri Lanka Employment Predictor.
Run this script to train the model from scratch.
"""
import argparse
import sys
from modules.data_preprocessing import DataPreprocessor
from modules.model_training import ModelTrainer
from modules.model_evaluation import ModelEvaluator
from config import RAW_DATA_PATH


def train_pipeline(data_path=None, tune_hyperparams=False, cv_folds=5, model_type='xgboost'):
    """
    Complete training pipeline from data to trained model.
    
    Args:
        data_path: Path to raw data file
        tune_hyperparams: Whether to perform hyperparameter tuning
        cv_folds: Number of cross-validation folds
        model_type: Type of model to train ('xgboost', 'random_forest', 'decision_tree', 
                   'gradient_boosting', 'naive_bayes', 'logistic_regression')
    
    Returns:
        Dictionary with model, metrics, and evaluation results
    """
    print("\n" + "="*70)
    print(" SRI LANKA EMPLOYMENT PREDICTOR - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Data Preprocessing
    print("STEP 1: DATA PREPROCESSING")
    print("-" * 70)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, df = preprocessor.prepare_data_pipeline(data_path)
    print("\n")
    
    # Step 2: Model Training
    print("STEP 2: MODEL TRAINING")
    print("-" * 70)
    trainer = ModelTrainer(model_type=model_type)
    print(f"Selected model: {trainer.model_name}")
    print()
    
    # Optional: Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = trainer.cross_validate(X_train, y_train, cv=cv_folds)
    print("\n")
    
    # Optional: Hyperparameter tuning
    if tune_hyperparams:
        print("Performing hyperparameter tuning...")
        print("WARNING: This may take several minutes depending on the parameter grid.")
        best_params, model = trainer.hyperparameter_tuning(X_train, y_train)
        print("\n")
    else:
        print("Training model with default parameters...")
        model = trainer.train_model(X_train, y_train)
        print("\n")
    
    # Save model
    trainer.save_model()
    print("\n")
    
    # Step 3: Model Evaluation
    print("STEP 3: MODEL EVALUATION")
    print("-" * 70)
    evaluator = ModelEvaluator(model, X_test, y_test)
    
    # Get feature names from preprocessor
    feature_names = preprocessor.feature_columns if hasattr(preprocessor, 'feature_columns') else None
    metrics = evaluator.evaluate_model(feature_names=feature_names)
    print("\n")
    
    # Summary
    print("="*70)
    print(" TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("\nSummary:")
    print(f"  Dataset size: {len(X_train) + len(X_test)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(preprocessor.feature_columns) if hasattr(preprocessor, 'feature_columns') else X_train.shape[1]}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score (weighted): {metrics['f1_weighted']:.4f}")
    
    if tune_hyperparams:
        print(f"\nBest hyperparameters found:")
        for param, value in trainer.best_params.items():
            print(f"    {param}: {value}")
    
    print("\nModel artifacts saved:")
    print("  - model.pkl")
    print("  - scaler.pkl")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - shap_summary.png")
    print("  - sample_dataset.csv")
    
    print("\nYou can now run the Streamlit app: streamlit run app.py")
    print("="*70 + "\n")
    
    return {
        'model': model,
        'trainer': trainer,
        'metrics': metrics,
        'cv_scores': cv_scores,
        'data': {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    }


def main():
    """Main function for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Train Sri Lanka Employment Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (XGBoost)
  python train_pipeline.py
  
  # Train with Random Forest
  python train_pipeline.py --model random_forest
  
  # Train with Decision Tree
  python train_pipeline.py --model decision_tree
  
  # Train with custom data file
  python train_pipeline.py --data path/to/labour_force_stats_sri_lanka.csv
  
  # Train with hyperparameter tuning
  python train_pipeline.py --tune
  
  # Train with 10-fold cross-validation
  python train_pipeline.py --cv 10
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help=f'Path to raw data CSV file (default: {RAW_DATA_PATH})'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest', 'decision_tree', 'gradient_boosting', 'naive_bayes', 'logistic_regression'],
        help='Model type to train (default: xgboost)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning (slower but may improve performance)'
    )
    
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        results = train_pipeline(
            data_path=args.data,
            tune_hyperparams=args.tune,
            cv_folds=args.cv,
            model_type=args.model
        )
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: Training pipeline failed!")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
