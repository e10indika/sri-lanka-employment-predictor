# Quick Start Guide - Employment Prediction

## ✅ Setup Complete!

Your project is now configured for **Sri Lankan Employment Prediction** with all your custom preprocessing logic integrated.

## 🚀 Next Steps

### Step 1: Train Your First Model

```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor
python3 train_pipeline.py
```

This will:
1. ✅ Load your `labour_force_stats_sri_lanka.csv` 
2. ✅ Create language profiles (SIN+ENG+TAMIL → encoded)
3. ✅ Combine employment columns → binary target
4. ✅ Aggregate disability features → severity categories
5. ✅ Train XGBoost binary classifier
6. ✅ Generate evaluation plots
7. ✅ Save model, scaler, and feature list

**Expected output:**
```
==================================================================
 SRI LANKA EMPLOYMENT PREDICTOR - TRAINING PIPELINE
==================================================================

STEP 1: DATA PREPROCESSING
----------------------------------------------------------------------
Loaded data with shape: (N, 24)
Engineering features for employment prediction...
Feature columns (X): [list of features]
Train set size: ...
Test set size: ...
Class distribution in train set:
  Unemployed (0): XXX
  Employed (1): XXX

STEP 2: MODEL TRAINING
----------------------------------------------------------------------
Performing 5-fold cross-validation...
Mean CV accuracy: 0.XXXX (+/- 0.XXXX)
Training XGBoost model for employment prediction...
Model training complete!

STEP 3: MODEL EVALUATION
----------------------------------------------------------------------
Accuracy:           0.XXXX
F1-Score (weighted): 0.XXXX
...
```

### Step 2: Launch the Web Interface

```bash
streamlit run app.py
```

Your browser will open to: `http://localhost:8501`

**Available pages:**
1. **Dashboard** - View model performance
2. **Dataset** - Browse your data  
3. **Predict** - Make predictions
4. **Train Model** - Interactive training

### Step 3: Make a Prediction

Navigate to **Predict** page and enter values for features like:
- SEX, AGE, MARITAL
- EDU, DEGREE
- Language_Profile_Encoded
- Disability_Category_Encoded
- DISTRICT
- etc.

Click "Predict Employment Status" to get:
- **Prediction**: Employed or Unemployed
- **Probabilities**: P(Unemployed), P(Employed)
- **SHAP Explanation**: Which features influenced the prediction

## 📊 What Was Implemented

### Your Preprocessing Logic ✅

**1. Language Profile**
```python
SIN=1, ENG=1, TAMIL=0 → "ENG+SIN" → Encoded value
```

**2. Employment Status**
```python
IF Employment=0 AND Employment_2=0 → Unemployed (0)
ELSE → Employed (1)
```

**3. Disability Features**
```python
6 disability columns (1-4 scale) →
  - Disability_Severity_Score (sum)
  - Disability_Category (None/Mild/Moderate/Severe)
  - Disability_Category_Encoded (0/1/2/3)
```

**4. Feature Exclusions**
```python
Excluded: SERNO, PSU, SECTOR, CUEDU, Unemployment Reason,
          Vocational Trained, Certified On Employment
```

### Model Configuration ✅

```python
Binary Classification (XGBoost)
- Objective: binary:logistic
- Target: 0=Unemployed, 1=Employed
- Features: Dynamically determined (~10-15 features)
- Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
```

## 📁 File Structure

```
sri-lanka-employment-predictor/
├── data/
│   └── labour_force_stats_sri_lanka.csv  ← Your data
├── modules/
│   ├── data_preprocessing.py   ← Your preprocessing logic
│   ├── model_training.py       ← Binary XGBoost
│   └── model_evaluation.py     ← Metrics & plots
├── pages/
│   ├── dashboard.py           ← Performance view
│   ├── dataset.py             ← Data viewer
│   ├── predict.py             ← Make predictions
│   └── train.py               ← Interactive training
├── config.py                  ← Configuration
├── train_pipeline.py          ← CLI training
├── app.py                     ← Streamlit entry
└── README.md                  ← Full documentation
```

## 🎯 Training Options

### Basic Training
```bash
python3 train_pipeline.py
```

### With Hyperparameter Tuning (slower, potentially better)
```bash
python3 train_pipeline.py --tune
```

### With Custom CV Folds
```bash
python3 train_pipeline.py --cv 10
```

### With Custom Data Path
```bash
python3 train_pipeline.py --data path/to/your/data.csv
```

## 📈 Expected Outputs

After training, you'll have:

1. **model.pkl** - Trained XGBoost model
2. **scaler.pkl** - StandardScaler for normalization
3. **feature_columns.json** - List of feature names
4. **confusion_matrix.png** - 2x2 matrix (Employed vs Unemployed)
5. **feature_importance.png** - Top predictors of employment
6. **shap_summary.png** - Feature contribution explanations
7. **sample_dataset.csv** - First 100 rows for UI

## 🔍 Interpreting Results

### Confusion Matrix
```
                Predicted
              Unemp  Empl
Actual Unemp  [ TP    FP ]
       Empl   [ FN    TN ]
```

### Feature Importance
Shows which factors most influence employment predictions:
- Education level
- Age
- Language capabilities
- Disability status
- etc.

### SHAP Values
Explains individual predictions:
- Red bars: Push toward Employed
- Blue bars: Push toward Unemployed

## ⚠️ Common Issues

**Issue**: Module import errors
```bash
# Ensure you're in project root
cd /Users/indika/External/MSc/sri-lanka-employment-predictor
python3 train_pipeline.py
```

**Issue**: "Feature columns not defined" in prediction page
- Train the model first to generate feature_columns.json

**Issue**: Missing data columns
- Verify your CSV has: SIN, ENG, TAMIL, Employment, Employment_2, and disability columns

## 📚 Documentation

- [README.md](README.md) - Full documentation
- [DATASET_MIGRATION.md](DATASET_MIGRATION.md) - Detailed migration notes
- [config.py](config.py) - All configuration options

## 🎉 Ready to Go!

Your employment prediction system is fully configured with:
✅ Custom preprocessing (language, employment, disability)
✅ Binary classification model
✅ Interactive web interface
✅ Comprehensive evaluation
✅ SHAP explainability

**Start training now:**
```bash
python3 train_pipeline.py
```

Then launch the app:
```bash
streamlit run app.py
```

Good luck with your employment prediction model! 🚀
