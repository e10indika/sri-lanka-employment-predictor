"""
Example script demonstrating how to use the modular components programmatically.
This shows how to train and evaluate an employment prediction model using the modules.
"""

from modules import DataPreprocessor, ModelTrainer, ModelEvaluator
import argparse


def main():
    """
    Example workflow for training and evaluating an employment prediction model.
    """
    parser = argparse.ArgumentParser(description='Example model training workflow')
    parser.add_argument('--data', type=str, default=None, 
                       help='Path to data file (optional)')
    args = parser.parse_args()
    
    print("Sri Lanka Employment Predictor - Example Usage\n")
    print("="*70)
    
    # Step 1: Preprocess data
    print("\n1. Preprocessing Data...")
    print("-"*70)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, df = preprocessor.prepare_data_pipeline(args.data)
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {len(preprocessor.feature_columns) if hasattr(preprocessor, 'feature_columns') else 'N/A'}")
    if 'Employment_Status_Encoded' in df.columns:
        print(f"  Classes: Unemployed (0), Employed (1)")
        print(f"  Employment rate: {(df['Employment_Status_Encoded'] == 1).mean():.2%}")
    
    # Step 2: Train model
    print("\n2. Training Model...")
    print("-"*70)
    trainer = ModelTrainer()
    
    # Cross-validation
    print("\nCross-validation:")
    cv_scores = trainer.cross_validate(X_train, y_train, cv=5)
    print(f"  Mean accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    
    # Train final model
    print("\nTraining final model...")
    model = trainer.train_model(X_train, y_train, verbose=False)
    trainer.save_model()
    print("  ✓ Model saved")
    
    # Step 3: Evaluate model
    print("\n3. Evaluating Model...")
    print("-"*70)
    evaluator = ModelEvaluator(model, X_test, y_test)
    feature_names = preprocessor.feature_columns if hasattr(preprocessor, 'feature_columns') else None
    metrics = evaluator.evaluate_model(feature_names=feature_names)
    
    # Step 4: Make example predictions
    print("\n4. Example Predictions...")
    print("-"*70)
    
    # Get 3 random samples
    import numpy as np
    sample_indices = np.random.choice(len(X_test), 3, replace=False)
    
    print("\nSample predictions:")
    for i, idx in enumerate(sample_indices, 1):
        X_sample = X_test[idx:idx+1]
        y_true = y_test.iloc[idx]
        y_pred = trainer.predict(X_sample)[0]
        status_true = "Employed" if y_true == 1 else "Unemployed"
        status_pred = "Employed" if y_pred == 1 else "Unemployed"
        
        print(f"  Sample {i}: True={status_true}, Predicted={status_pred} "
              f"{'✓' if y_true == y_pred else '✗'}")
    
    print("\n" + "="*70)
    print("Example workflow complete!")
    print("="*70)
    
    print("\nNext steps:")
    print("  - Run 'streamlit run app.py' to launch the web interface")
    print("  - Check generated visualizations (*.png files)")
    print("  - Modify config.py to adjust model parameters")
    

if __name__ == "__main__":
    main()
