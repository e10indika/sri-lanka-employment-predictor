"""
Model training module for Sri Lanka Employment Predictor.
Handles model initialization, training, hyperparameter tuning, and saving.
"""
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import numpy as np
import os
from config import MODEL_PATH, MODEL_PARAMS, CV_FOLDS, RANDOM_STATE, MODEL_CONFIGS, DEFAULT_MODEL_TYPE


class ModelTrainer:
    """Handles model training and hyperparameter optimization for multiple ML models."""
    
    def __init__(self, model_type=None, model_params=None):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model ('xgboost', 'random_forest', 'decision_tree', 
                       'gradient_boosting', 'naive_bayes', 'logistic_regression')
            model_params: Dictionary of model hyperparameters. 
                         If None, uses default params from config.
        """
        if model_type is None:
            model_type = DEFAULT_MODEL_TYPE
        
        self.model_type = model_type
        
        if model_params is None:
            if model_type in MODEL_CONFIGS:
                model_params = MODEL_CONFIGS[model_type]['params']
                self.model_name = MODEL_CONFIGS[model_type]['name']
            else:
                model_params = MODEL_PARAMS
                self.model_name = 'XGBoost'
        else:
            self.model_name = model_type.replace('_', ' ').title()
        
        self.model_params = model_params
        self.model = None
        self.best_params = None
    
    def create_model(self, params=None):
        """
        Create classifier with given parameters based on model_type.
        
        Args:
            params: Model parameters dictionary
        
        Returns:
            Classifier instance
        """
        if params is None:
            params = self.model_params
        
        if self.model_type == 'xgboost':
            model = xgb.XGBClassifier(**params)
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(**params)
        elif self.model_type == 'decision_tree':
            model = DecisionTreeClassifier(**params)
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(**params)
        elif self.model_type == 'naive_bayes':
            model = GaussianNB(**params)
        elif self.model_type == 'logistic_regression':
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def train_model(self, X_train, y_train, verbose=True):
        """
        Train the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Whether to print training progress
        
        Returns:
            Trained model
        """
        print(f"Training {self.model_name} model for employment prediction...")
        
        self.model = self.create_model()
        
        # For binary classification, labels should be 0 and 1
        # Ensure proper format
        y_train_adjusted = y_train.astype(int)
        
        # XGBoost supports verbose parameter, others don't
        if self.model_type == 'xgboost' and verbose:
            self.model.fit(X_train, y_train_adjusted, verbose=verbose)
        else:
            self.model.fit(X_train, y_train_adjusted)
        
        print(f"{self.model_name} training complete!")
        print(f"Classes: 0=Unemployed, 1=Employed")
        
        return self.model
    
    def cross_validate(self, X_train, y_train, cv=None):
        """
        Perform cross-validation on the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
        
        Returns:
            Cross-validation scores
        """
        if cv is None:
            cv = CV_FOLDS
        
        print(f"Performing {cv}-fold cross-validation...")
        
        model = self.create_model()
        y_train_adjusted = y_train.astype(int)
        
        scores = cross_val_score(
            model, X_train, y_train_adjusted, 
            cv=cv, scoring='accuracy'
        )
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None, cv=3):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary with parameters names as keys and lists of values
            cv: Number of cross-validation folds
        
        Returns:
            Best parameters and best model
        """
        if param_grid is None:
            # Default parameter grid for tuning
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        
        print("Starting hyperparameter tuning...")
        print(f"Parameter grid: {param_grid}")
        
        base_model = self.create_model()
        y_train_adjusted = y_train.astype(int)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train_adjusted)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return self.best_params, self.model
    
    def save_model(self, filepath=None, save_model_info=True):
        """
        Save trained model to disk with model-specific filename in models/ directory.
        
        Args:
            filepath: Path to save model. If None, uses model type in filename.
            save_model_info: Whether to save model metadata (type, name, etc.)
        """
        if self.model is None:
            raise ValueError("No trained model to save. Train a model first.")
        
        # If no filepath provided, create one with model type in models/ directory
        if filepath is None:
            from config import MODEL_DIR
            filepath = os.path.join(MODEL_DIR, f'model_{self.model_type}.pkl')
        
        # Save model data
        model_data = {
            'model': self.model,
            'params': self.model_params,
            'model_type': self.model_type,
            'model_name': self.model_name
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
        # Save model info for the UI
        if save_model_info:
            from config import MODEL_INFO_PATH
            import json
            model_info = {
                'model_type': self.model_type,
                'model_name': self.model_name,
                'model_path': filepath
            }
            with open(MODEL_INFO_PATH, 'w') as f:
                json.dump(model_info, f)
            print(f"Model info saved to {MODEL_INFO_PATH}")
    
    def load_model(self, filepath=None):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to model file
        
        Returns:
            Loaded model
        """
        if filepath is None:
            filepath = MODEL_PATH
        
        model_data = joblib.load(filepath)
        
        # Handle both old and new model formats
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.model_params = model_data.get('params', MODEL_PARAMS)
        else:
            # Old format - just the model
            self.model = model_data
            self.model_params = MODEL_PARAMS
        
        print(f"Model loaded from {filepath}")
        return self.model
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions (0=Unemployed, 1=Employed)
        """
        if self.model is None:
            raise ValueError("No trained model available. Train or load a model first.")
        
        predictions = self.model.predict(X)
        
        return predictions.astype(int)


if __name__ == "__main__":
    # Test the trainer
    from modules.data_preprocessing import DataPreprocessor
    
    print("Testing Model Trainer...")
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data_pipeline()
    
    # Train model
    trainer = ModelTrainer()
    trainer.train_model(X_train, y_train)
    
    # Cross-validate
    trainer.cross_validate(X_train, y_train)
    
    # Save model
    trainer.save_model()
    
    print("\nModel training test complete!")
