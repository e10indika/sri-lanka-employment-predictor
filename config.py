"""
Configuration file for the Sri Lanka Employment Predictor project.
Contains paths, model parameters, and feature definitions.
"""
import os

# Get base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'labour_force_stats_sri_lanka.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

# Model paths (organized in models/ directory)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')  # Legacy/default model path
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.json')  # Stores which model is currently active
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, 'feature_columns.json')

# Visualization paths (organized in visualizations/ directory)
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')
# Legacy paths (for backward compatibility)
CM_PATH = os.path.join(VIZ_DIR, 'confusion_matrix.png')
FI_PATH = os.path.join(VIZ_DIR, 'feature_importance.png')
SHAP_PATH = os.path.join(VIZ_DIR, 'shap_summary.png')

# Functions to get model-specific visualization paths
def get_model_viz_paths(model_type):
    """
    Get visualization paths for a specific model.
    
    Args:
        model_type: Model type (e.g., 'xgboost', 'decision_tree')
    
    Returns:
        Dictionary with paths for confusion_matrix, feature_importance, and shap_summary
    """
    return {
        'confusion_matrix': os.path.join(VIZ_DIR, f'confusion_matrix_{model_type}.png'),
        'feature_importance': os.path.join(VIZ_DIR, f'feature_importance_{model_type}.png'),
        'shap_summary': os.path.join(VIZ_DIR, f'shap_summary_{model_type}.png')
    }

# Sample data (organized in data/ directory)
SAMPLE_PATH = os.path.join(DATA_DIR, 'sample_dataset.csv')
FEATURE_INFO_PATH = os.path.join(DATA_DIR, 'feature_info.json')

# Feature definitions - will be determined dynamically after preprocessing
# Initial columns in raw data (before preprocessing)
RAW_COLUMNS = [
    'SECTOR', 'DISTRICT', 'PSU', 'SERNO', 'SEX', 'AGE', 'MARITAL', 'EDU', 
    'DEGREE', 'CUEDU', 'SIN', 'ENG', 'TAMIL', 'Eye Disability', 
    'Hearing Disability', 'Walking Disability', 'Remembering Disability', 
    'Self Care Disability', 'Communicating Disability', 'Vocational Trained',
    'Employment', 'Employment_2', 'Unemployment Reason', 'Certified On Employment'
]

# Columns to exclude from features
COLUMNS_TO_EXCLUDE = [
    'SERNO', 'PSU', 'SECTOR', 'CUEDU', 'CUEDU_log', 'DEGREE',
    'Unemployment Reason', 'Vocational Trained', 'Certified On Employment',
    'Language_Profile', 'Disability_Category', 'Employment_Status_Categorical',
    'Employment', 'Employment_2'
]

TARGET_COLUMN = 'Employment_Status_Encoded'

# Feature columns will be set dynamically after preprocessing
FEATURE_COLUMNS = []  # Populated after preprocessing

# Function to save/load feature columns
def save_feature_columns(feature_cols):
    """Save feature columns to a file for later use."""
    import json
    with open(FEATURE_COLUMNS_PATH, 'w') as f:
        json.dump(feature_cols, f)
    print(f"Feature columns saved to {FEATURE_COLUMNS_PATH}")

def load_feature_columns():
    """Load feature columns from file if available."""
    import json
    if os.path.exists(FEATURE_COLUMNS_PATH):
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            return json.load(f)
    return []

def get_available_models():
    """Get list of available trained models from models/ directory."""
    import glob
    model_files = glob.glob(os.path.join(MODEL_DIR, 'model_*.pkl'))
    available_models = []
    
    for model_file in model_files:
        try:
            import joblib
            model_data = joblib.load(model_file)
            if isinstance(model_data, dict) and 'model_type' in model_data:
                model_type = model_data['model_type']
                viz_paths = get_model_viz_paths(model_type)
                available_models.append({
                    'path': model_file,
                    'filename': os.path.basename(model_file),
                    'model_type': model_type,
                    'model_name': model_data.get('model_name', model_type),
                    'visualizations': viz_paths
                })
        except:
            pass
    
    return available_models

# Try to load existing feature columns
FEATURE_COLUMNS = load_feature_columns()

# Default model type
DEFAULT_MODEL_TYPE = 'xgboost'  # Options: 'xgboost', 'random_forest', 'decision_tree', 'gradient_boosting', 'naive_bayes', 'logistic_regression'

# Model hyperparameters for different models
MODEL_CONFIGS = {
    'xgboost': {
        'name': 'XGBoost',
        'params': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': 1
        }
    },
    'random_forest': {
        'name': 'Random Forest',
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    },
    'decision_tree': {
        'name': 'Decision Tree',
        'params': {
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'random_state': 42
        }
    },
    'naive_bayes': {
        'name': 'Naive Bayes',
        'params': {}
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'params': {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }
    }
}

# Legacy support - default to XGBoost params
MODEL_PARAMS = MODEL_CONFIGS['xgboost']['params']

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)
