# Sri Lanka Employment Predictor

A modular machine learning application for predicting employment status with support for multiple ML models, comprehensive model training, evaluation, and deployment capabilities based on Sri Lankan labour force statistics.

## Features

- 🤖 **Multiple ML Models**: Support for 6 models (XGBoost, Random Forest, Decision Tree, Gradient Boosting, Naive Bayes, Logistic Regression)
- 🎯 **Modular Training Pipeline**: Complete modular training system with model comparison
- 📊 **Interactive Dashboard**: Streamlit-based web interface with model selection
- 🔮 **Prediction System**: Real-time employment status predictions with selected model
- 📈 **Model Explainability**: SHAP-based feature importance and explanations for all models
- 🎨 **Visualizations**: Confusion matrix, feature importance, and SHAP plots
- 📁 **Organized Structure**: Clean separation of data, models, modules, and frontend

## Project Structure

```
├── app.py                      # Main Streamlit application entry point
├── config.py                   # Central configuration (paths, model configs)
├── utils.py                    # Utility functions for model loading/predictions
├── requirements.txt            # Python dependencies
├── train_pipeline.py          # Complete training pipeline script
├── PROJECT_STRUCTURE.md       # Detailed structure documentation
│
├── data/                       # Data files (organized)
│   ├── labour_force_stats_sri_lanka.csv
│   ├── processed_data.csv
│   ├── sample_dataset.csv
│   └── feature_info.json
│
├── models/                     # Trained models and artifacts (organized)
│   ├── model_xgboost.pkl
│   ├── model_decision_tree.pkl
│   ├── model_*.pkl            # Other trained models
│   ├── scaler.pkl
│   ├── model_info.json
│   └── feature_columns.json
│
├── modules/                    # Backend logic (organized)
│   ├── data_preprocessing.py  # Data loading, cleaning, feature engineering
│   ├── model_training.py      # Model training and tuning
│   └── model_evaluation.py    # Evaluation and visualization
│
├── pages/                      # Frontend pages (Streamlit convention)
│   ├── dashboard.py           # Model performance dashboard
│   ├── dataset.py             # Dataset exploration
│   ├── train.py               # Model training interface
│   ├── predict.py             # Prediction interface
│   └── compare_models.py      # Model comparison
│
└── visualizations/            # Generated plots (organized)
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── shap_summary.png
```

**See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation.**
│
├── data/                       # Data directory (created on first run)
│   └── winequality-red.csv    # Raw dataset
│
└── models/                     # Model artifacts (created during training)
    ├── model.pkl              # Trained model
    ├── scaler.pkl             # Feature scaler
    ├── confusion_matrix.png   # Performance visualizations
    ├── feature_importance.png
    └── shap_summary.png
```

## Setup and Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/sri-lanka-employment-predictor.git
cd sri-lanka-employment-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Place your `labour_force_stats_sri_lanka.csv` file in the `data/` directory, or use the training interface to upload your dataset.

## Usage

### Option 1: Train Model via Command Line

Train the model with default settings:
```bash
python train_pipeline.py
```

Train with custom data file:
```bash
python train_pipeline.py --data path/to/labour_force_stats_sri_lanka.csv
```

Train with hyperparameter tuning:
```bash
python train_pipeline.py --tune
```

Train with 10-fold cross-validation:
```bash
python train_pipeline.py --cv 10
```

### Option 2: Train Model via Web Interface

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Navigate to the "Train Model" page
3. Upload your dataset or use the default path
4. Configure training options in the sidebar
5. Click "Start Training" and monitor progress

### Option 3: Use Pre-trained Model

If you already have a trained model, simply place `model.pkl` and `scaler.pkl` in the project root directory and run the Streamlit app.

## Training Pipeline Details

The training pipeline consists of three main stages:

### 1. Data Preprocessing (`modules/data_preprocessing.py`)
- Load CSV data with employment statistics
- Handle missing values and duplicates
- Feature engineering:
  - Language profile creation from language columns (SIN, ENG, TAMIL)
  - Employment status combination from Employment and Employment_2
  - Disability features aggregation and categorization
- Train-test split with stratification
- Feature binary scaling using StandardScaler

### 2. Model Training (`modules/model_training.py`)
- XGBoost classifier initialization
- Cross-validation for performance estimation
- Optional hyperparameter tuning with GridSearchCV
- Model persistence to disk

### 3. Model Evaluation (`modules/model_evaluation.py`)
- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Feature importance analysis
- SHAP-based global and local explainability

## Streamlit Application Pagesemployment datasets
3. **Predict**: Make employment status predictions on individual record
1. **Dashboard**: View model performance metrics and visualizations
2. **Dataset**: Upload, view, and download datasets
3. **Predict**: Make predictions on individual wine samples
4. **Train Model**: Interactive model training interface

## Model Configuration

Edit `config.py` to customize:
- Data paths
- Model hyperparameters
- Feature definitions
- Training parameters

Example model parameters:
```python
MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'objective': 'binary:logistic'  # Binary classification
}
```

## Dataset Requirements

The CSV file should contain columns such as:
- **SECTOR, DISTRICT, PSU, SERNO**: Geographic/survey identifiers
- **SEX, AGE, MARITAL**: Demographics
- **EDU, DEGREE, CUEDU**: Education level
- **SIN, ENG, TAMIL**: Language proficiency (0/1 indicators)
- **Eye Disability, Hearing Disability, Walking Disability, Remembering Disability, Self Care Disability, Communicating Disability**: Disability indicators (1-4 scale)
- **Vocational Trained**: Vocational training status
- **Employment, Employment_2**: Employment status indicators
- **Unemployment Reason**: Reason for unemployment (if applicable)
- **Certified On Employment**: Employment certification status

Delimiter: comma (`,`)

### Feature Engineering

The preprocessing pipeline automatically creates:
- **Language_Profile_Encoded**: Combined language capabilities
- **Employment_Status_Encoded**: Binary employment status (0=Unemployed, 1=Employed)
- **Disability_Category_Encoded**: Categorized disability severity

## Development and Testing

### Test Individual Modules

Test data preprocessing:
```bash
python modules/data_preprocessing.py
```

Test model training:
```bash
python modules/model_training.py
```

Test model evaluation:
```bash
python modules/model_evaluation.py
```

### Run Streamlit Pages Independently

While the pages are designed for the multi-page app, you can debug individual pages:
```bash
streamlit run pages/train.py
```

## Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Set `app.py` as the entry point
4. Deploy!

Note: Ensure model files are generated or uploaded before deployment.

### Docker task: Binary classification (Employed vs Unemployed)
- Typical accuracy: Varies by dataset characteristics and class balance
- Key features: Education level, age, language proficiency, disability status

## Performance Notes

- Expected accuracy: 0.65-0.70 on test set
- Top features: alcohol content, volatile acidity
- Training time: 1-3 minutes (without tuning)
- Tuning time: 5-15 minutes (with GridSearchCV)

## Troubleshooting

**Issue**: "File not found" errors
- Ensure dataset is in `data/` directory
- Run training pipeline to generate model files

**Issue**: Import errors in modules
- Check that you're running from project root
- Verify all dependencies are installed

**Issue**: Streamlit pages not showing
- Ensure pages are in `pages/` directory
- Check that files are named correctly (`.py` extension)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

##Dataset: Sri Lankan Labour Force Statistics

MIT License - feel free to use and modify as needed.

## Acknowledgments

- Wine Quality Dataset: UCI Machine Learning Repository
- XGBoost: Gradient boosting framework
- SHAP: Model explainability library
- Streamlit: Interactive web framework