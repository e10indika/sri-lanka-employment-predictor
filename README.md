# Sri Lanka Employment Predictor

Machine learning system for predicting employment status in Sri Lanka using socio-demographic and disability-related factors.

## Architecture

This project consists of two independent applications:

### 🔧 Backend API
**Location:** `/backend`  
**Technology:** FastAPI + Python  
**Port:** 8000

RESTful API server providing:
- Model training and management
- Predictions (single & batch)
- Dataset statistics
- Model comparison
- Background job processing

[→ Backend Documentation](backend/README.md)

### 🎨 Frontend Application
**Location:** `/frontend`  
**Technology:** React + Material-UI  
**Port:** 3000

Modern web interface featuring:
- Interactive dashboards
- Model visualization
- Training interface
- Prediction forms
- Model comparison charts

[→ Frontend Documentation](frontend/README.md)

## Quick Start

### Option 1: Run Both Servers

**Terminal 1 - Backend:**
```bash
cd backend
pip install -r requirements.txt
./start_server.sh
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open: http://localhost:3000

### Option 2: Standalone Backend Only

```bash
cd backend
pip install -r requirements.txt
./start_server.sh
```

API Documentation: http://localhost:8000/docs

### Option 3: Standalone Frontend Only

```bash
cd frontend
npm install
npm run dev
```

Configure backend URL in `frontend/.env`:
```env
VITE_API_URL=https://your-backend-api.com
```

## Project Structure

```
sri-lanka-employment-predictor/
│
├── backend/                 # FastAPI Backend (Independent)
│   ├── api/
│   │   ├── main.py         # FastAPI app
│   │   └── routes/         # API endpoints
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                # React Frontend (Independent)
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Page views
│   │   └── services/       # API client
│   ├── package.json
│   └── README.md
│
├── modules/                 # Shared ML Modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── data/                    # Dataset files
├── models/                  # Trained models (.pkl)
├── visualizations/          # Generated plots
├── config.py               # Shared configuration
└── README.md              # This file
```

## Deployment Scenarios

### Production Deployment: GitHub Pages + Railway

**Recommended setup for free hosting:**

**Frontend → GitHub Pages (Free)**
- Deployment: `npm run deploy`
- URL: `https://yourusername.github.io/sri-lanka-employment-predictor`
- Cost: Free
- [→ Frontend Deployment Guide](frontend/README.md#deployment)

**Backend → Railway/Render (Free Tier)**
- Deployment: `railway up` or Render dashboard
- URL: `https://your-app.railway.app`
- Cost: Free (Railway: $5 credit/month)
- [→ Backend Deployment Guide](backend/README.md#deployment)

**📖 Complete Deployment Guide:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### Quick Deployment

```bash
# 1. Deploy Backend (Railway)
cd backend
npm install -g @railway/cli
railway login
railway init
railway up
# Note the URL: https://your-app.railway.app

# 2. Deploy Frontend (GitHub Pages)
cd ../frontend
npm install --save-dev gh-pages
# Add to package.json: "homepage": "https://yourusername.github.io/reponame"
# Add to scripts: "deploy": "gh-pages -d dist"
# Update .env.production: VITE_API_URL=https://your-app.railway.app
npm run deploy
```

Visit: `https://yourusername.github.io/reponame`

### Scenario 1: Separate Deployments

**Backend:**
- Deploy to: AWS Lambda, Google Cloud Run, Heroku
- Endpoint: `https://api.your-domain.com`

**Frontend:**
- Deploy to: Netlify, Vercel, AWS S3 + CloudFront, **GitHub Pages**
- Endpoint: `https://your-domain.com`

**Configuration:**
```env
# frontend/.env
VITE_API_URL=https://api.your-domain.com
```

### Scenario 2: Same Server (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /var/www/frontend/dist;
        try_files $uri /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

### Scenario 3: Docker Compose

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    environment:
      - VITE_API_URL=http://backend:8000
```

## Features

### Machine Learning
- Multiple model support (XGBoost, Random Forest, Decision Tree, Logistic Regression)
- Automated training with cross-validation
- Hyperparameter tuning
- Model evaluation metrics
- SHAP explanations

### API Features
- RESTful endpoints
- Automatic documentation (Swagger UI)
- Background job processing
- CORS enabled
- File upload support

### Frontend Features  
- Responsive Material-UI design
- Real-time training progress
- Interactive visualizations (Recharts)
- Model comparison dashboard
- Batch predictions
- Export capabilities

## Documentation

- **[Backend API Documentation](backend/README.md)** - API setup, endpoints, deployment
- **[Frontend Documentation](frontend/README.md)** - React app setup, components, deployment
- **[React + FastAPI Architecture Guide](REACT_FASTAPI_ARCHITECTURE.md)** - System design
- **[Model Comparison Feature](MODEL_COMPARISON_FEATURE.md)** - Comparison feature docs
- **[Quick Start Guide](START_SERVERS.md)** - Getting started

## Development Workflow

### Backend Development

```bash
cd backend
pip install -r requirements.txt

# Run with auto-reload
cd api
uvicorn main:app --reload --port 8000

# Test endpoints
curl http://localhost:8000/api/models/
```

### Frontend Development

```bash
cd frontend
npm install

# Run dev server with hot reload
npm run dev

# Build for production
npm run build
```

### Adding New Features

**Backend - New Endpoint:**
1. Create route in `backend/api/routes/`
2. Add to `main.py`
3. Document in OpenAPI

**Frontend - New Page:**
1. Create component in `frontend/src/pages/`
2. Add route in `App.jsx`
3. Add to navigation in `Layout.jsx`

## API Endpoints

### Base URL
`http://localhost:8000`

### Main Endpoints
- `GET /api/models/` - List models
- `GET /api/models/compare` - Compare models
- `POST /api/training/start` - Train model
- `POST /api/predictions/predict` - Predict
- `GET /api/datasets/info` - Dataset stats

Full API documentation: http://localhost:8000/docs

## Environment Configuration

### Backend `.env`
```env
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

### Frontend `.env`
```env
VITE_API_URL=http://localhost:8000
```

## Testing

### Backend
```bash
cd backend
pytest  # If tests are implemented
```

### Frontend
```bash
cd frontend
npm test  # If tests are implemented
```

### Manual API Testing
```bash
# Using curl
curl http://localhost:8000/api/models/

# Using httpie
http GET localhost:8000/api/models/

# Using Postman
Import: http://localhost:8000/openapi.json
```

## Troubleshooting

### Backend Issues

**Port 8000 in use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Module import errors:**
```bash
# Run from project root
cd /path/to/sri-lanka-employment-predictor
python -m uvicorn backend.api.main:app --reload
```

### Frontend Issues

**API connection failed:**
1. Check backend is running on port 8000
2. Verify CORS configuration in backend
3. Check `VITE_API_URL` in `.env`

**Build errors:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Cross-Origin Issues

Backend `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Data Privacy & Security

- No authentication implemented (add for production)
- Dataset should be sanitized before use
- CORS configured for development (restrict in production)
- Secure API keys and credentials
- Validate all user inputs

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes in either `backend/` or `frontend/`
4. Test both independently
5. Submit pull request

## Technology Stack Summary

### Backend
- FastAPI, Uvicorn, Pydantic
- scikit-learn, XGBoost, pandas, numpy
- matplotlib, seaborn, SHAP
- Python 3.10+

### Frontend
- React 18, React Router 6
- Material-UI 5, Emotion
- Axios, Recharts
- Vite, Node.js 18+

## License

[Your License Here]

## Support

- Backend Issues: See [backend/README.md](backend/README.md)
- Frontend Issues: See [frontend/README.md](frontend/README.md)
- General: Open an issue in the repository

---

**Project Status:** Production Ready  
**Last Updated:** January 15, 2026  
**Architecture:** Microservices (Separated Frontend & Backend)

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