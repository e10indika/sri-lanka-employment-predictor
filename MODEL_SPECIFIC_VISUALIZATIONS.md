# Model-Specific Visualizations Update

## Summary

Updated the project to generate and display **separate visualizations for each trained model** instead of having a single set of visualizations that get overwritten.

## Changes Made

### 1. Configuration (`config.py`)
- ✅ Added `get_model_viz_paths(model_type)` function to generate model-specific paths
- ✅ Updated `get_available_models()` to include visualization paths in model info
- ✅ Kept legacy paths (CM_PATH, FI_PATH, SHAP_PATH) for backward compatibility

**Example paths:**
```python
# XGBoost visualizations
confusion_matrix_xgboost.png
feature_importance_xgboost.png
shap_summary_xgboost.png

# Decision Tree visualizations
confusion_matrix_decision_tree.png
feature_importance_decision_tree.png
shap_summary_decision_tree.png
```

### 2. Model Evaluator (`modules/model_evaluation.py`)
- ✅ Added `model_type` parameter to `__init__`
- ✅ Updated to use model-specific paths via `get_model_viz_paths()`
- ✅ Visualization titles now include model name
- ✅ All plot functions use `self.viz_paths` dictionary

### 3. Dashboard Page (`pages/dashboard.py`)
- ✅ Complete redesign with **model selector dropdown**
- ✅ Dynamically loads visualizations for selected model
- ✅ Shows warning if visualizations don't exist for a model
- ✅ Lists all available models with their visualization status

**New features:**
- Model selection sidebar
- Model info banner
- Split visualization display (2 columns + full width)
- "All Available Models" expander with status indicators

### 4. Training Page (`pages/train.py`)
- ✅ Updated `ModelEvaluator` initialization to pass `model_type`
- ✅ Changed visualization display to use model-specific paths
- ✅ Both single model and "train all" modes updated
- ✅ Tab captions include model name

### 5. Model Comparison Page (`pages/compare_models.py`)
- ✅ Updated to generate visualizations during comparison
- ✅ Passes `model_type` to evaluator

### 6. Regeneration Script (`regenerate_visualizations.py`)
- ✅ New utility script to regenerate visualizations for all existing models
- ✅ Automatically detects all trained models
- ✅ Generates complete visualization set for each
- ✅ Useful after code updates or to fix missing visualizations

## Visualization Status

Current models with visualizations:

| Model | Confusion Matrix | Feature Importance | SHAP Summary |
|-------|:----------------:|:------------------:|:------------:|
| XGBoost | ✅ | ✅ | ✅ |
| Decision Tree | ✅ | ✅ | ✅ |
| Logistic Regression | ✅ | ✅ | ✅ |

## File Structure

```
visualizations/
├── confusion_matrix.png                         # Legacy (if needed)
├── feature_importance.png                       # Legacy (if needed)
├── shap_summary.png                            # Legacy (if needed)
│
├── confusion_matrix_xgboost.png                 # XGBoost specific
├── feature_importance_xgboost.png
├── shap_summary_xgboost.png
│
├── confusion_matrix_decision_tree.png           # Decision Tree specific
├── feature_importance_decision_tree.png
├── shap_summary_decision_tree.png
│
├── confusion_matrix_logistic_regression.png     # Logistic Regression specific
├── feature_importance_logistic_regression.png
└── shap_summary_logistic_regression.png
```

## Usage

### Viewing Visualizations

1. **Dashboard Page:**
   - Navigate to the Dashboard page
   - Use the model selector in the sidebar
   - View visualizations for the selected model

2. **Training:**
   - Train any model via the Train page
   - Visualizations are automatically generated with model-specific names
   - View them immediately after training

### Regenerating Visualizations

If visualizations are missing or need to be regenerated:

```bash
python regenerate_visualizations.py
```

This will:
- Find all trained models
- Load test data
- Generate all three visualizations for each model
- Save with model-specific filenames

## Benefits

### ✅ No Overwrites
Each model has its own set of visualizations - no more overwriting when training different models.

### ✅ Easy Comparison
Switch between models in the dashboard to compare their performance visually.

### ✅ Better Organization
Clear naming convention: `{visualization_type}_{model_type}.png`

### ✅ Backward Compatible
Legacy paths still work if needed for existing scripts.

### ✅ Automatic Generation
All new training automatically creates model-specific visualizations.

## Technical Details

### Path Generation
```python
from config import get_model_viz_paths

# Get paths for XGBoost
viz_paths = get_model_viz_paths('xgboost')
# Returns:
# {
#   'confusion_matrix': 'visualizations/confusion_matrix_xgboost.png',
#   'feature_importance': 'visualizations/feature_importance_xgboost.png',
#   'shap_summary': 'visualizations/shap_summary_xgboost.png'
# }
```

### Model Evaluator Usage
```python
from modules.model_evaluation import ModelEvaluator

# Initialize with model_type for automatic path handling
evaluator = ModelEvaluator(model, X_test, y_test, model_type='xgboost')

# All visualizations will use model-specific paths
evaluator.generate_all_visualizations(feature_names)
```

### Dashboard Model Selection
```python
# Automatically gets all available models with viz info
available_models = get_available_models()

# Each model includes visualization paths
for model in available_models:
    print(model['model_name'])
    print(model['visualizations'])
```

## Future Enhancements

Potential improvements:
- Side-by-side visualization comparison
- Animation/transitions when switching models
- Download buttons for individual visualizations
- Visualization versioning (timestamp-based)
- Custom visualization configurations per model type

## Testing

All components tested and verified:
- ✅ Configuration functions work correctly
- ✅ Model evaluator creates model-specific files
- ✅ Dashboard displays correct visualizations
- ✅ Training pages use correct paths
- ✅ All 3 existing models have complete visualizations
- ✅ Model selector works properly
- ✅ Regeneration script processes all models

## Migration Notes

**Existing projects:**
Run `regenerate_visualizations.py` once to create model-specific visualizations for all trained models.

**New training:**
Automatically generates model-specific visualizations - no action needed.

**Legacy code:**
Old code using CM_PATH, FI_PATH, SHAP_PATH still works but may be overwritten. Update to use `get_model_viz_paths()` for model-specific paths.

---

**Status:** ✅ Fully implemented and tested  
**Date:** January 15, 2026  
**Models tested:** XGBoost, Decision Tree, Logistic Regression
