# How to Start the Application

## Quick Start

### Terminal 1 - Backend (FastAPI)
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor
/Users/indika/venvs/py3kernel/bin/python -m uvicorn backend.api.main:app --reload --port 8000
```

**Backend will run at:** http://localhost:8000
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Terminal 2 - Frontend (React)
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor/frontend

# First time only: install dependencies
npm install

# Start dev server
npm run dev
```

**Frontend will run at:** http://localhost:3000

## Detailed Steps

### 1. Backend Setup (First Time)
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor/backend
/Users/indika/venvs/py3kernel/bin/pip install -r requirements.txt
```

### 2. Frontend Setup (First Time)
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor/frontend
npm install
```

### 3. Start Backend
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor
/Users/indika/venvs/py3kernel/bin/python -m uvicorn backend.api.main:app --reload --port 8000
```

Keep this terminal running.

### 4. Start Frontend (New Terminal)
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor/frontend
npm run dev
```

Keep this terminal running.

### 5. Access the Application

Open your browser and go to: **http://localhost:3000**

## Troubleshooting

### Backend Issues

**Problem:** Module not found errors
```bash
# Make sure you're in the project root, not in backend/api
cd /Users/indika/External/MSc/sri-lanka-employment-predictor
/Users/indika/venvs/py3kernel/bin/python -m uvicorn backend.api.main:app --reload --port 8000
```

**Problem:** Port 8000 already in use
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9
```

**Problem:** Missing dependencies
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor/backend
/Users/indika/venvs/py3kernel/bin/pip install -r requirements.txt
```

### Frontend Issues

**Problem:** npm: command not found
```bash
# Install Node.js from https://nodejs.org/ (version 18+)
# Or using Homebrew:
brew install node
```

**Problem:** Port 3000 already in use
The dev server will ask if you want to use a different port. Type 'y'.

**Problem:** Dependencies not installed
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor/frontend
rm -rf node_modules package-lock.json
npm install
```

### CORS Issues

If frontend can't connect to backend:
1. Make sure backend is running on port 8000
2. Check backend logs for errors
3. Verify CORS is configured in [backend/api/main.py](backend/api/main.py)

## Production Deployment

### Backend
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor
/Users/indika/venvs/py3kernel/bin/python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
cd /Users/indika/External/MSc/sri-lanka-employment-predictor/frontend
npm run build
# Serve the dist/ folder with nginx or any static server
```

## Architecture Overview

```
┌────────────────────────────────────────┐
│   React Frontend (Port 3000)          │
│   http://localhost:3000                │
└────────────────┬───────────────────────┘
                 │ REST API
                 ▼
┌────────────────────────────────────────┐
│   FastAPI Backend (Port 8000)         │
│   http://localhost:8000                │
│                                        │
│   • /api/models/                      │
│   • /api/training/                    │
│   • /api/predictions/                 │
│   • /api/datasets/                    │
│   • /api/visualizations/              │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│   Shared Python Modules                │
│   • data_preprocessing                 │
│   • model_training                     │
│   • model_evaluation                   │
└────────────────────────────────────────┘
```

## Testing the Backend API

### Using curl
```bash
# Test health check
curl http://localhost:8000/

# List models
curl http://localhost:8000/api/models/

# Get dataset info
curl http://localhost:8000/api/datasets/info

# Make a prediction
curl -X POST http://localhost:8000/api/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "features": {
      "DISTRICT": 1,
      "SEX": 1,
      "AGE": 25,
      "MARITAL": 1,
      "EDU": 3,
      "Language_Profile_Encoded": 1,
      "Disability_Category_Encoded": 0
    }
  }'
```

### Using Swagger UI
Open http://localhost:8000/docs in your browser and test endpoints interactively.

## Next Steps

1. ✅ Backend dependencies installed
2. ✅ Backend running on port 8000
3. ⏳ Install frontend dependencies: `cd frontend && npm install`
4. ⏳ Start frontend dev server: `npm run dev`
5. ⏳ Open http://localhost:3000 in browser

## Documentation

- Full Architecture Guide: [REACT_FASTAPI_ARCHITECTURE.md](REACT_FASTAPI_ARCHITECTURE.md)
- Project README: [README.md](README.md)
- API Documentation: http://localhost:8000/docs (when backend is running)
