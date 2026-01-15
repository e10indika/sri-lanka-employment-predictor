# Project Separation Guide

This document explains how the Sri Lanka Employment Predictor has been organized into two independent projects.

## Overview

The monolithic application has been separated into:

1. **Backend API** (`/backend`) - Standalone FastAPI server
2. **Frontend App** (`/frontend`) - Standalone React application

Both can be developed, deployed, and scaled independently.

## Directory Structure

### Before (Monolithic)
```
sri-lanka-employment-predictor/
├── app.py                    # Streamlit app (deprecated)
├── pages/                    # Streamlit pages (deprecated)
├── requirements.txt          # Mixed dependencies
└── utils.py                  # Mixed utilities
```

### After (Separated)
```
sri-lanka-employment-predictor/
├── backend/                  # ✅ Independent Backend
│   ├── api/
│   │   ├── main.py          # FastAPI entry point
│   │   └── routes/          # API endpoints
│   ├── requirements.txt     # Backend-only dependencies
│   ├── start_server.sh      # Backend startup
│   └── README.md            # Backend documentation
│
├── frontend/                 # ✅ Independent Frontend
│   ├── src/
│   │   ├── pages/           # React pages
│   │   ├── components/      # React components
│   │   └── services/        # API client
│   ├── package.json         # Frontend-only dependencies
│   └── README.md            # Frontend documentation
│
└── modules/                  # 🔄 Shared (Backend dependencies)
    ├── data_preprocessing.py
    ├── model_training.py
    └── model_evaluation.py
```

## Key Changes

### Backend

**Location:** `/backend`

**What's Included:**
- ✅ FastAPI application
- ✅ API routes (models, training, predictions, datasets, visualizations)
- ✅ Background task processing
- ✅ CORS middleware
- ✅ API documentation (Swagger/ReDoc)

**What's Excluded:**
- ❌ Frontend code
- ❌ Node.js dependencies
- ❌ React components
- ❌ Streamlit legacy code

**Dependencies:**
```
backend/requirements.txt:
- fastapi
- uvicorn
- scikit-learn
- xgboost
- pandas
- matplotlib
```

**Startup:**
```bash
cd backend
./start_server.sh
# Runs on: http://localhost:8000
```

### Frontend

**Location:** `/frontend`

**What's Included:**
- ✅ React 18 application
- ✅ Material-UI components
- ✅ React Router for navigation
- ✅ Axios API client
- ✅ Recharts for visualizations
- ✅ Vite build system

**What's Excluded:**
- ❌ Backend code
- ❌ Python dependencies
- ❌ ML models
- ❌ Data processing

**Dependencies:**
```
frontend/package.json:
- react
- react-router-dom
- @mui/material
- axios
- recharts
- vite
```

**Startup:**
```bash
cd frontend
npm run dev
# Runs on: http://localhost:3000
```

## Communication Between Projects

### API Integration

The frontend communicates with the backend via REST API:

**Frontend → Backend:**
```javascript
// frontend/src/services/api.js
const API_BASE_URL = 'http://localhost:8000';

export const modelsAPI = {
  getAll: () => axios.get(`${API_BASE_URL}/api/models/`),
  compare: () => axios.get(`${API_BASE_URL}/api/models/compare`),
};
```

**Backend → Frontend:**
```python
# backend/api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Environment Configuration

**Backend `.env`:**
```env
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

**Frontend `.env`:**
```env
VITE_API_URL=http://localhost:8000
```

## Deployment Options

### Option 1: Separate Servers (Recommended)

**Advantages:**
- Independent scaling
- Different hosting providers
- Isolated failures
- Technology flexibility

**Setup:**

1. **Deploy Backend:**
   - AWS Lambda / Google Cloud Run / Heroku
   - URL: `https://api.yourdomain.com`

2. **Deploy Frontend:**
   - Netlify / Vercel / AWS S3 + CloudFront
   - URL: `https://yourdomain.com`

3. **Connect:**
   ```env
   # Frontend .env
   VITE_API_URL=https://api.yourdomain.com
   ```

### Option 2: Single Server (Nginx)

**Advantages:**
- Simple configuration
- Single domain
- Lower cost

**Setup:**

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Serve frontend
    location / {
        root /var/www/frontend/dist;
        try_files $uri /index.html;
    }

    # Proxy API requests to backend
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Option 3: Docker Compose

**Advantages:**
- Consistent environment
- Easy local development
- Production-like setup

**Setup:**

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    environment:
      - VITE_API_URL=http://backend:8000
    depends_on:
      - backend
```

Run:
```bash
docker-compose up
```

## Development Workflow

### Working on Backend Only

```bash
cd backend
pip install -r requirements.txt
./start_server.sh

# Test endpoints
curl http://localhost:8000/api/models/
```

### Working on Frontend Only

```bash
cd frontend
npm install
npm run dev

# Frontend connects to existing backend
# (can be local or remote)
```

### Working on Both

**Terminal 1:**
```bash
cd backend
./start_server.sh
```

**Terminal 2:**
```bash
cd frontend
npm run dev
```

## Shared Resources

### Modules (Backend Dependency)

Location: `/modules`

Used by: Backend only

```python
# Backend can import directly
from modules.data_preprocessing import DataPreprocessor
from modules.model_training import ModelTrainer
from modules.model_evaluation import ModelEvaluator
```

Frontend accesses functionality via API endpoints.

### Data Directory

Location: `/data`

Used by: Backend for training/predictions

```
data/
├── raw_data.csv              # Original dataset
├── processed_data.csv        # Preprocessed
└── sample_dataset.csv        # Sample for display
```

### Models Directory

Location: `/models`

Used by: Backend for loading/saving models

```
models/
├── model_xgboost.pkl
├── model_random_forest.pkl
├── feature_columns.json
└── scaler.pkl
```

### Visualizations Directory

Location: `/visualizations`

Used by: Backend generates, Frontend displays

```
visualizations/
├── confusion_matrix_xgboost.png
├── feature_importance_xgboost.png
└── shap_summary_xgboost.png
```

## Benefits of Separation

### 1. Independent Development
- Backend team works on Python/FastAPI
- Frontend team works on React/TypeScript
- No conflicts or dependencies

### 2. Independent Deployment
- Deploy backend updates without frontend changes
- Deploy frontend updates without backend changes
- Different release cycles

### 3. Technology Flexibility
- Replace frontend (React → Vue/Angular)
- Replace backend (FastAPI → Django/Flask)
- Without affecting the other

### 4. Scalability
- Scale backend horizontally for ML workload
- Scale frontend via CDN
- Optimize each independently

### 5. Multiple Frontends
- Web app (React)
- Mobile app (React Native)
- Desktop app (Electron)
- All using same backend API

### 6. Testing
- Unit test backend independently
- E2E test frontend independently
- Integration test API contracts

## Migration Checklist

- [x] Backend separated to `/backend`
- [x] Frontend separated to `/frontend`
- [x] Backend has own README
- [x] Frontend has own README
- [x] Backend has own dependencies (requirements.txt)
- [x] Frontend has own dependencies (package.json)
- [x] API client configured in frontend
- [x] CORS configured in backend
- [x] Environment variables documented
- [x] Startup scripts created
- [x] Documentation updated
- [x] Both projects tested independently

## Troubleshooting

### Backend can't find modules

**Issue:** `ImportError: No module named 'modules'`

**Solution:** Run from project root:
```bash
cd /path/to/sri-lanka-employment-predictor
python -m uvicorn backend.api.main:app --reload
```

### Frontend can't connect to backend

**Issue:** CORS error or connection refused

**Solutions:**
1. Check backend is running: `curl http://localhost:8000`
2. Check CORS origins in `backend/api/main.py`
3. Verify `VITE_API_URL` in `frontend/.env`

### Port conflicts

**Issue:** Port already in use

**Solutions:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Change backend port in start_server.sh
# Change frontend port in vite.config.js
```

## Next Steps

1. **Add Authentication**
   - JWT tokens in backend
   - Token storage in frontend
   - Protected routes

2. **Add Testing**
   - Backend: pytest
   - Frontend: Vitest + React Testing Library
   - Integration: Cypress

3. **Add CI/CD**
   - GitHub Actions
   - Separate pipelines for backend/frontend
   - Automated deployments

4. **Add Monitoring**
   - Backend: Prometheus + Grafana
   - Frontend: Google Analytics / Sentry
   - API performance tracking

5. **Add Documentation**
   - API versioning
   - Changelog
   - Migration guides

## Resources

- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)
- [Architecture Guide](REACT_FASTAPI_ARCHITECTURE.md)
- [Quick Start](START_SERVERS.md)

---

**Separation Status:** ✅ Complete  
**Architecture:** Microservices  
**Communication:** REST API  
**Last Updated:** January 15, 2026
