# Sri Lanka Employment Predictor - Backend API

FastAPI-based machine learning service for predicting employment status in Sri Lanka using socio-demographic and disability-related factors.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
./start_server.sh
```

API will be available at:
- **API Server:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## ✨ Features

### Machine Learning
- **Multiple Models:** LightGBM (primary), XGBoost, Random Forest, and more
- **Automated Training:** Background job processing with progress tracking
- **Hyperparameter Tuning:** 4-level tuning system (none, basic, default grid, custom grid)
- **Cross-Validation:** K-fold validation with configurable folds
- **Model Persistence:** Automatic saving with SHAP explainers

### Model Interpretability & Analysis
- **Feature Importance:** Global feature ranking and contribution analysis
- **SHAP Analysis:** Game-theory based feature attribution with summary plots
- **LIME Explanations:** Individual prediction explanations for stakeholders
- **Partial Dependence Plots:** Visualize feature effects on predictions
- **Comprehensive Visualizations:** Auto-generated during training

### API Capabilities
- **RESTful API:** Clean, documented endpoints
- **Batch Predictions:** Process multiple predictions at once
- **Model Comparison:** Side-by-side performance metrics
- **Dataset Analysis:** Statistics, samples, correlations
- **Data Preprocessing:** Endpoint for processing raw CSV files
- **Visualizations:** 5 types of plots including confusion matrix, feature importance, SHAP, LIME, and PDP

### Performance
- **Fast Inference:** Optimized prediction pipeline
- **Async Processing:** Non-blocking training operations
- **CORS Enabled:** Ready for external access
- **Network Access:** Configured for `0.0.0.0` (all interfaces)

## 🏗️ Architecture

```
sri-lanka-employment-predictor/
├── api/                       # FastAPI application
│   ├── main.py               # FastAPI app entry point
│   └── routes/               # API route handlers
│       ├── models.py         # Model management
│       ├── training.py       # Training endpoints
│       ├── predictions.py    # Prediction endpoints
│       ├── datasets.py       # Dataset endpoints
│       └── visualizations.py # Visualization endpoints
├── modules/                   # ML pipeline modules
│   ├── data_preprocessing.py # Data cleaning and feature engineering
│   ├── model_training.py     # Model training logic
│   └── model_evaluation.py   # Evaluation metrics
├── data/                      # Datasets
├── models/                    # Trained model artifacts
├── visualizations/            # Generated plots
├── config.py                 # Configuration settings
├── train_pipeline.py         # Standalone training script
├── start_server.sh           # Server startup script
└── requirements.txt          # Dependencies
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB RAM minimum
- 500MB disk space

### Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd sri-lanka-employment-predictor
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt

## 🔧 Running the Server

### Development Mode

```bash
cd backend
./start_server.sh
```

Or manually:
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
cd n:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🌐 API Endpoints

### Models
- `GET /api/models/` - List all trained models
- `GET /api/models/configs` - Get available model configurations
- `GET /api/models/{model_type}/details` - Get specific model details
- `GET /api/models/compare` - Compare all models
- `DELETE /api/models/{model_type}` - Delete a model

### Training
- `POST /api/training/start` - Start training a new model
- `GET /api/training/status/{job_id}` - Get training job status
- `GET /api/training/jobs` - List all training jobs
- `DELETE /api/training/jobs/{job_id}` - Delete a training job

### Predictions
- `POST /api/predictions/predict` - Make a single prediction
- `POST /api/predictions/batch-predict` - Batch predictions
- `GET /api/predictions/features` - Get feature information

### Datasets
- `GET /api/datasets/info` - Dataset statistics
- `GET /api/datasets/sample` - Get sample rows
- `GET /api/datasets/column/{name}` - Column information
- `GET /api/datasets/correlation` - Correlation matrix

### Visualizations
- `GET /api/visualizations/{model}/confusion-matrix` - Confusion matrix image
- `GET /api/visualizations/{model}/feature-importance` - Feature importance plot
- `GET /api/visualizations/{model}/shap-summary` - SHAP summary plot
- `GET /api/visualizations/{model}/status` - Visualization status

## 🎓 Model Training

### Using the API

```bash
curl -X POST "http://localhost:8000/api/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "perform_cv": true,
    "cv_folds": 5,
    "perform_tuning": true
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/training/start",
    json={
        "model_type": "xgboost",
        "perform_cv": True,
        "cv_folds": 5,
        "perform_tuning": True
    }
)

job_id = response.json()["job_id"]
print(f"Training started: {job_id}")
```

### Available Models

- `xgboost` - XGBoost Classifier (Recommended)
- `random_forest` - Random Forest Classifier
- `gradient_boosting` - Gradient Boosting Classifier
- `logistic_regression` - Logistic Regression
- `svm` - Support Vector Machine
- `neural_network` - Multi-layer Perceptron

## 🔮 Making Predictions

### Single Prediction

```bash
curl -X POST "http://localhost:8000/api/predictions/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "features": {
      "DISTRICT": 1,
      "SEX": 1,
      "AGE": 35,
      "MARITAL": 2,
      "EDU": 3,
      "Language Profile": 1,
      "Disability": 0
    }
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predictions/predict",
    json={
        "model_type": "xgboost",
        "features": {
            "DISTRICT": 1,
            "SEX": 1,
            "AGE": 35,
            "MARITAL": 2,
            "EDU": 3,
            "Language Profile": 1,
            "Disability": 0
        }
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
```

## 📊 Dataset Features

1. **DISTRICT** (1-25): Geographic district code
2. **SEX** (1-2): Gender (1=Male, 2=Female)
3. **AGE** (15-65): Age in years
4. **MARITAL** (1-5): Marital status
5. **EDU** (1-10): Education level
6. **Language Profile** (1-3): Primary language
7. **Disability** (0-1): Disability status

**Target:** EMPLOYED (0=Not Employed, 1=Employed)

## 🚢 External Access

### Local Network

The server binds to `0.0.0.0:8000`, making it accessible on your local network.

Find your local IP:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Access from other devices:
```
http://YOUR_LOCAL_IP:8000
```

### Internet Access (ngrok)

```bash
# Install ngrok
brew install ngrok

# Expose backend
ngrok http 8000
```

### Cloud Deployment

**Railway:**
```bash
npm install -g @railway/cli
cd backend
railway login
railway init
railway up
```

**Render / Fly.io / Vercel:** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 🛠️ Development

### API Documentation

Once running, access interactive docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Project Structure

- **backend/api/routes/** - API endpoints
- **** - ML pipeline (training, evaluation, preprocessing)
- **models/** - Saved model files (.pkl)
- **data/** - Dataset files
- **visualizations/** - Generated plots

## 🔒 Security

### CORS Configuration

Update `backend/api/main.py` for production:
```pytho
allow_origins=["https://yourfrontend.com"]  # Restrict origins
```

### Recommendations

1. Add API authentication (JWT/API keys)
2. Implement rate limiting
3. Use HTTPS in production
4. Validate all inputs
5. Enable monitoring and logging

## 🐛 Troubleshooting

### Port Already in Use
```bash
lsof -i :8000
kill -9 $(lsof -t -i:8000)
```

### Module Import Errors
```bash
pip install -r requirements.txt --force-reinstall
```

### Model Not Found
```bash
python train_pipeline.py --model xgboost
```

## 👨‍💻 Frontend Application

The frontend UI has been separated into its own project:

**Location:** `/Users/indika/External/GitHub/sri-lanka-employment-predictor-UI`

```bash
cd /Users/indika/External/GitHub/sri-lanka-employment-predictor-UI
npm install
npm run dev
```

Frontend will be available at: http://localhost:3000

## 📚 Additional Documentation

- [Backend API Details](backend/README.md)
- [Model Visualizations](MODEL_SPECIFIC_VISUALIZATIONS.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Quick Start API](QUICK_START_API.md)
---

**Version:** 1.0.0  
**Last Updated:** January 16, 2026  
**Status:** Production Ready ✅
