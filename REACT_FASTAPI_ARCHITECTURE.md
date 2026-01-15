# React + FastAPI Architecture

## Overview

The Sri Lanka Employment Predictor has been transformed from a Streamlit monolith into a modern **React frontend + FastAPI backend** architecture with complete separation of concerns.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                      │
│          (Port 3000 - Development)                   │
│                                                      │
│  • Material-UI Components                           │
│  • React Router for navigation                      │
│  • Axios for API calls                             │
│  • Vite for fast development                        │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP/REST API
                   │
┌──────────────────▼──────────────────────────────────┐
│              FastAPI Backend                         │
│             (Port 8000 - API Server)                 │
│                                                      │
│  • RESTful API endpoints                            │
│  • Background job processing                        │
│  • Model training & predictions                     │
│  • Dataset management                               │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│          Shared Core Modules                         │
│                                                      │
│  • modules/data_preprocessing.py                    │
│  • modules/model_training.py                        │
│  • modules/model_evaluation.py                      │
│  • config.py                                        │
└─────────────────────────────────────────────────────┘
```

## Directory Structure

```
sri-lanka-employment-predictor/
│
├── backend/                      # FastAPI Backend Server
│   ├── api/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── routes/
│   │   │   ├── models.py        # Model management endpoints
│   │   │   ├── training.py      # Training endpoints
│   │   │   ├── predictions.py   # Prediction endpoints
│   │   │   ├── datasets.py      # Dataset endpoints
│   │   │   └── visualizations.py # Visualization endpoints
│   │   └── middleware/           # Custom middleware
│   ├── requirements.txt          # Backend dependencies
│   └── start_server.sh           # Server startup script
│
├── frontend/                     # React Frontend Application
│   ├── public/                   # Static assets
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   └── Layout.jsx       # Main layout with navigation
│   │   ├── pages/               # Page components
│   │   │   ├── Dashboard.jsx    # Model dashboard
│   │   │   ├── DatasetView.jsx  # Dataset explorer
│   │   │   ├── TrainModel.jsx   # Training interface
│   │   │   ├── Predict.jsx      # Prediction interface
│   │   │   └── CompareModels.jsx # Model comparison
│   │   ├── services/
│   │   │   └── api.js           # API client & services
│   │   ├── App.jsx              # Main app component
│   │   ├── main.jsx             # Entry point
│   │   └── index.css            # Global styles
│   ├── package.json             # Frontend dependencies
│   ├── vite.config.js           # Vite configuration
│   └── index.html               # HTML template
│
├── modules/                      # Shared Python Modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── data/                         # Data files
├── models/                       # Trained models
├── visualizations/               # Generated plots
├── config.py                     # Shared configuration
└── README.md                     # Updated documentation
```

## API Endpoints

### Models API (`/api/models`)
- `GET /` - List all trained models
- `GET /configs` - Get model configurations
- `GET /{model_type}` - Get model info
- `GET /{model_type}/details` - Get detailed model info
- `DELETE /{model_type}` - Delete a model

### Training API (`/api/training`)
- `POST /start` - Start training job
- `GET /status/{job_id}` - Get training status
- `GET /jobs` - List all training jobs
- `DELETE /jobs/{job_id}` - Delete training job

### Predictions API (`/api/predictions`)
- `POST /predict` - Make single prediction
- `POST /batch-predict` - Make batch predictions
- `GET /features` - Get feature schema

### Datasets API (`/api/datasets`)
- `GET /info` - Get dataset information
- `GET /sample` - Get sample data
- `GET /column/{column_name}` - Get column info
- `GET /correlation` - Get correlation matrix

### Visualizations API (`/api/visualizations`)
- `GET /{model_type}/confusion-matrix` - Get confusion matrix
- `GET /{model_type}/feature-importance` - Get feature importance
- `GET /{model_type}/shap-summary` - Get SHAP summary
- `GET /{model_type}/status` - Get visualization status

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **scikit-learn, XGBoost** - ML libraries
- **Pandas, NumPy** - Data processing
- **Matplotlib, Seaborn, SHAP** - Visualizations

### Frontend
- **React 18** - UI library
- **Material-UI (MUI)** - Component library
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **Vite** - Build tool & dev server
- **Recharts** - Data visualization (optional)

## Running the Application

### Prerequisites
```bash
# Python 3.10+ with virtual environment
# Node.js 18+ with npm
```

### Backend Setup
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start server (development)
chmod +x start_server.sh
./start_server.sh

# Or manually
cd api
uvicorn main:app --reload --port 8000
```

Backend will run at: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run at: `http://localhost:3000`

### Production Build
```bash
# Backend
cd backend/api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Frontend
cd frontend
npm run build
# Serve the dist/ folder with nginx or any static server
```

## Development Workflow

### 1. Backend Development
- API routes in `backend/api/routes/`
- Shared logic in `modules/`
- Test endpoints at `http://localhost:8000/docs`

### 2. Frontend Development
- React components in `frontend/src/components/`
- Pages in `frontend/src/pages/`
- API calls in `frontend/src/services/api.js`
- Hot reload enabled via Vite

### 3. Testing API
```bash
# Using curl
curl http://localhost:8000/api/models/

# Using httpie
http GET localhost:8000/api/models/

# Using Swagger UI
# Open http://localhost:8000/docs in browser
```

## Key Features

### Backend
- ✅ RESTful API design
- ✅ Background job processing for training
- ✅ CORS enabled for frontend
- ✅ Automatic API documentation
- ✅ Model-specific visualizations
- ✅ Batch predictions
- ✅ Error handling & validation

### Frontend
- ✅ Modern React with hooks
- ✅ Material-UI components
- ✅ Responsive design
- ✅ Real-time training progress
- ✅ Model selection & comparison
- ✅ Interactive predictions
- ✅ Dataset exploration

## API Client Example

```javascript
// frontend/src/services/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Get all models
const models = await axios.get(`${API_BASE_URL}/api/models/`);

// Train a model
const response = await axios.post(`${API_BASE_URL}/api/training/start`, {
  model_type: 'xgboost',
  perform_cv: true,
  cv_folds: 5
});

// Make prediction
const prediction = await axios.post(`${API_BASE_URL}/api/predictions/predict`, {
  model_type: 'xgboost',
  features: {
    DISTRICT: 1,
    SEX: 1,
    AGE: 25,
    // ... other features
  }
});
```

## Benefits of New Architecture

### 1. Separation of Concerns
- Frontend handles UI/UX
- Backend handles business logic
- Clear API contract

### 2. Scalability
- Frontend and backend can scale independently
- Can deploy to different servers
- Add load balancers easily

### 3. Flexibility
- Replace frontend (React → Vue, Angular)
- Replace backend (FastAPI → Flask, Django)
- Multiple frontends (web, mobile)

### 4. Development Speed
- Parallel frontend/backend development
- Hot reload on both sides
- Clear API documentation

### 5. Production Ready
- Can deploy to cloud (AWS, Azure, GCP)
- Docker containerization ready
- CI/CD pipeline friendly

## Migration from Streamlit

### What Changed
- **Streamlit pages** → **React pages**
- **st.button()** → **Material-UI buttons**
- **st.dataframe()** → **MUI Tables**
- **Direct function calls** → **REST API calls**
- **Session state** → **React state + API**

### What Stayed
- All core ML logic (`modules/`)
- Data preprocessing pipeline
- Model training & evaluation
- Visualization generation
- Configuration system

## Future Enhancements

- [ ] WebSocket for real-time training updates
- [ ] User authentication & authorization
- [ ] Model versioning system
- [ ] A/B testing support
- [ ] Monitoring & logging dashboard
- [ ] Docker Compose setup
- [ ] Kubernetes deployment configs
- [ ] GraphQL API alternative
- [ ] Mobile app (React Native)
- [ ] Automated testing (Jest, Pytest)

## Troubleshooting

### CORS Issues
If frontend can't connect to backend:
```python
# In backend/api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Port Conflicts
- Backend default: 8000 (change in start_server.sh)
- Frontend default: 3000 (change in vite.config.js)

### Import Errors
Backend can't find modules:
```bash
# Make sure you're in the right directory
cd backend/api
python -m uvicorn main:app --reload
```

## Documentation

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Frontend**: http://localhost:3000

## Contact & Support

See main README.md for project details and contribution guidelines.

---

**Architecture Status:** ✅ Fully implemented  
**Backend:** FastAPI + Python  
**Frontend:** React + Material-UI  
**Date:** January 15, 2026
