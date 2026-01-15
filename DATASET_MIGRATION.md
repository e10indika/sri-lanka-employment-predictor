# Dataset Migration Summary

## Successfully Updated: Wine Quality → Sri Lanka Employment

Your project has been successfully migrated from wine quality prediction to employment prediction using Sri Lankan labour force statistics.

## What Changed

### 1. **Configuration** ([config.py](config.py))
- ✅ Dataset path: `labour_force_stats_sri_lanka.csv`
- ✅ Model objective: Binary classification (Employed vs Unemployed)
- ✅ Target column: `Employment_Status_Encoded`
- ✅ Feature columns: Dynamically determined after preprocessing
- ✅ Added feature column persistence (saved to `feature_columns.json`)

### 2. **Data Preprocessing** ([modules/data_preprocessing.py](modules/data_preprocessing.py))
Implemented your custom preprocessing logic:

#### Language Profile Creation
```python
# Combines SIN, ENG, TAMIL into encoded profiles
df['Language_Profile'] = create_language_profile(row)
df['Language_Profile_Encoded'] = label_encoder.fit_transform(df['Language_Profile'])
```

#### Employment Status
```python
# Combines Employment and Employment_2 columns
# Creates binary target: 0=Unemployed, 1=Employed
df['Employment_Status_Encoded'] = ...
```

#### Disability Features
```python
# Aggregates 6 disability columns into:
- Disability_Count
- Disability_Severity_Score
- Max_Disability_Severity
- Disability_Category (None/Mild/Moderate/Severe)
- Disability_Category_Encoded (0/1/2/3)
```

#### Feature Selection
Excludes as specified:
- Administrative: SERNO, PSU, SECTOR
- Education: CUEDU, CUEDU_log
- Other: Unemployment Reason, Vocational Trained, Certified On Employment
- Intermediate: Language_Profile, Disability_Category, Employment_Status_Categorical

### 3. **Model Training** ([modules/model_training.py](modules/model_training.py))
- ✅ Binary classification (no label offset needed)
- ✅ Classes: 0=Unemployed, 1=Employed
- ✅ Updated cross-validation
- ✅ Updated hyperparameter tuning

### 4. **Model Evaluation** ([modules/model_evaluation.py](modules/model_evaluation.py))
- ✅ Binary classification metrics
- ✅ Updated confusion matrix (2x2)
- ✅ Feature importance for employment predictors
- ✅ SHAP explanations

### 5. **Web Interface**
Updated all pages for employment context:

#### [app.py](app.py)
- Title: "Sri Lanka Employment Predictor"

#### [pages/predict.py](pages/predict.py)
- Dynamic feature inputs based on trained model
- Employment status prediction (Employed/Unemployed)
- Probability display for both classes
- SHAP explanations

#### [pages/dashboard.py](pages/dashboard.py)
- Employment-specific insights
- Binary classification visualizations

#### [pages/dataset.py](pages/dataset.py)
- Employment dataset viewer
- Employment rate statistics

#### [pages/train.py](pages/train.py)
- Employment data upload
- Progress tracking
- Binary classification training

### 6. **Documentation**
- ✅ [README.md](README.md): Full employment prediction documentation
- ✅ [train_pipeline.py](train_pipeline.py): CLI training for employment model
- ✅ [example_usage.py](example_usage.py): Updated examples

## How to Use

### 1. Train the Model
```bash
python train_pipeline.py
```

This will:
1. Load `data/labour_force_stats_sri_lanka.csv`
2. Apply your preprocessing logic automatically
3. Train binary XGBoost classifier
4. Generate evaluation visualizations
5. Save model artifacts

### 2. Run the Web App
```bash
streamlit run app.py
```

### 3. Make Predictions

The prediction page will dynamically show all features from your trained model. Just enter values and get:
- Employment status prediction (Employed/Unemployed)
- Probabilities for each class
- SHAP explanation showing which features influenced the prediction

## Key Features

### Automatic Feature Engineering
All your preprocessing steps run automatically:
- ✅ Language profile encoding
- ✅ Employment status combination
- ✅ Disability feature aggregation
- ✅ Proper feature exclusion

### Dynamic Feature Handling
- Feature columns are determined after preprocessing
- Saved to `feature_columns.json` for consistency
- Prediction page adapts to trained model features

### Binary Classification
- Model: XGBoost with `binary:logistic` objective
- Output: 0 (Unemployed) or 1 (Employed)
- Includes probability estimates

## Files Generated After Training

```
├── model.pkl                    # Trained XGBoost model
├── scaler.pkl                   # StandardScaler for features
├── feature_columns.json         # List of feature names
├── confusion_matrix.png         # 2x2 confusion matrix
├── feature_importance.png       # Top employment predictors
├── shap_summary.png            # SHAP explainability
├── sample_dataset.csv          # First 100 rows for UI
└── data/
    └── processed_data.csv      # Fully preprocessed dataset
```

## Feature Columns (After Preprocessing)

After running preprocessing, you'll have features like:
- Demographics: SEX, AGE, MARITAL, DISTRICT
- Education: EDU, DEGREE
- Language: Language_Profile_Encoded
- Disability: Disability_Category_Encoded
- And more based on what remains after exclusions

## Testing

```bash
# Test preprocessing only
python modules/data_preprocessing.py

# Test full pipeline
python example_usage.py

# Train with options
python train_pipeline.py --tune --cv 10
```

## Troubleshooting

**Issue**: "Feature columns not yet defined"
- **Solution**: Train the model first to generate feature columns

**Issue**: "Target column not found"
- **Solution**: Ensure Employment and Employment_2 columns exist in data

**Issue**: "Missing columns"
- **Solution**: Check that CSV has all required columns (SIN, ENG, TAMIL, disability columns, etc.)

## Next Steps

1. ✅ Your data is in `data/labour_force_stats_sri_lanka.csv`
2. ✅ Run `python train_pipeline.py` to train
3. ✅ Run `streamlit run app.py` to use the web interface
4. ✅ Check generated visualizations
5. ✅ Make predictions on new data

---

**Your employment prediction model is ready to train! 🚀**
