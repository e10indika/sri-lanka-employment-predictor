# Project Structure

## Overview
This document describes the organized structure of the Sri Lanka Employment Predictor project, with clear separation between data, models, modules, and frontend components.

## Directory Structure

```
sri-lanka-employment-predictor/
│
├── app.py                          # Main Streamlit application entry point
├── config.py                       # Central configuration file
├── utils.py                        # Utility functions for model loading and predictions
├── requirements.txt                # Python dependencies
├── README.md                       # Main project documentation
│
├── data/                           # Data files (organized)
│   ├── labour_force_stats_sri_lanka.csv    # Raw dataset
│   ├── processed_data.csv                   # Preprocessed data
│   ├── sample_dataset.csv                   # Sample data for testing
│   └── feature_info.json                    # Feature metadata
│
├── models/                         # Trained models and artifacts (organized)
│   ├── model_xgboost.pkl                    # XGBoost model
│   ├── model_decision_tree.pkl              # Decision Tree model
│   ├── model_logistic_regression.pkl        # Logistic Regression model
│   ├── model_random_forest.pkl              # Random Forest model (when trained)
│   ├── model_gradient_boosting.pkl          # Gradient Boosting model (when trained)
│   ├── model_naive_bayes.pkl                # Naive Bayes model (when trained)
│   ├── scaler.pkl                           # Feature scaler
│   ├── model_info.json                      # Currently active model metadata
│   └── feature_columns.json                 # Feature column names
│
├── modules/                        # Backend logic (organized)
│   ├── __init__.py                          # Module initialization
│   ├── data_preprocessing.py                # Data preprocessing pipeline
│   ├── model_training.py                    # Model training functionality
│   └── model_evaluation.py                  # Model evaluation and metrics
│
├── pages/                          # Frontend pages (Streamlit convention)
│   ├── dashboard.py                         # Dashboard with overview
│   ├── dataset.py                           # Dataset exploration
│   ├── train.py                             # Model training interface
│   ├── predict.py                           # Prediction interface
│   └── compare_models.py                    # Model comparison
│
└── visualizations/                 # Generated visualizations (organized)
    ├── confusion_matrix.png                 # Confusion matrix plot
    ├── feature_importance.png               # Feature importance plot
    └── shap_summary.png                     # SHAP summary plot
```

## Component Descriptions

### Root Level

- **app.py**: Main entry point for the Streamlit application
- **config.py**: Central configuration with paths, model configs, and hyperparameters
- **utils.py**: Shared utility functions for model loading, predictions, and visualizations
- **requirements.txt**: All Python package dependencies

### data/
Contains all data files:
- Raw datasets
- Processed/cleaned data
- Sample datasets for testing
- Feature metadata

### models/
Contains all trained models and model-related artifacts:
- Trained model files (*.pkl) with model type suffix
- Scaler for feature normalization
- Model metadata and configuration
- Feature column information

### modules/
Backend logic organized as reusable modules:
- **data_preprocessing.py**: Data loading, cleaning, feature engineering
- **model_training.py**: Model initialization, training, hyperparameter tuning
- **model_evaluation.py**: Performance metrics, visualization generation

### pages/
Frontend components following Streamlit's multi-page app structure:
- **dashboard.py**: Overview and project summary
- **dataset.py**: Data exploration and statistics
- **train.py**: Interactive model training interface
- **predict.py**: Make predictions with selected model
- **compare_models.py**: Side-by-side model comparison

### visualizations/
Generated plots and visualizations:
- Performance metrics plots
- Feature importance charts
- SHAP explanations
- Confusion matrices

## Benefits of This Structure

### 1. **Separation of Concerns**
- Data files isolated in `data/`
- Models isolated in `models/`
- Business logic in `modules/`
- UI components in `pages/`

### 2. **Scalability**
- Easy to add new models to `models/`
- Easy to add new modules to `modules/`
- Easy to add new pages to `pages/`

### 3. **Maintainability**
- Clear file organization
- Easy to locate specific components
- Logical grouping of related files

### 4. **Version Control**
- Can easily `.gitignore` large files in `models/` and `data/`
- Keep visualizations separate for easy exclusion

### 5. **Deployment Ready**
- Clean structure suitable for production deployment
- Easy to configure different paths for different environments
- Modular design supports testing and CI/CD

## Path Configuration

All paths are centrally managed in `config.py`:

```python
# Data paths
DATA_DIR = 'data/'
RAW_DATA_PATH = 'data/labour_force_stats_sri_lanka.csv'

# Model paths
MODEL_DIR = 'models/'
MODEL_PATH = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'

# Visualization paths
VIZ_DIR = 'visualizations/'
```

## Usage

### Training a Model
```bash
# Models are automatically saved to models/ directory
streamlit run app.py
# Navigate to "Train Model" page
```

### Running Predictions
```bash
# Select model from models/ directory
streamlit run app.py
# Navigate to "Predict" page
```

### Adding New Components

**New Model:**
1. Add configuration to `MODEL_CONFIGS` in `config.py`
2. Update `model_training.py` if needed
3. Train and it will save to `models/model_{type}.pkl`

**New Page:**
1. Create new file in `pages/` directory
2. Streamlit automatically detects it

**New Module:**
1. Create new Python file in `modules/`
2. Import and use in other components

## Best Practices

1. **Always use config.py paths** - Never hardcode file paths
2. **Keep data/ out of version control** - Large datasets should be .gitignored
3. **Organize models by type** - Use naming convention: `model_{type}.pkl`
4. **Document new features** - Update this file when structure changes
5. **Modular design** - Keep modules focused on single responsibilities

## Migration Notes

Files were reorganized as follows:
- `*.pkl` → `models/`
- `*.csv` (sample) → `data/`
- `*.png` → `visualizations/`
- `*_info.json` → `models/`

All code has been updated to use the new paths via `config.py`.
