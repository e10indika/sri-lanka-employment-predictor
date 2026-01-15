# Sri Lanka Employment Predictor - Backend API

FastAPI backend server for the Sri Lanka Employment Predictor application.

## Overview

RESTful API server that provides machine learning model training, predictions, and dataset management for employment prediction in Sri Lanka.

## Technology Stack

- **FastAPI 0.128+** - Modern Python web framework
- **Uvicorn** - ASGI server
- **scikit-learn 1.4+** - Machine learning library
- **XGBoost 2.0+** - Gradient boosting library
- **Pandas & NumPy** - Data processing
- **Matplotlib, Seaborn, SHAP** - Visualizations

## Project Structure

```
backend/
├── api/
│   ├── main.py              # FastAPI application
│   ├── routes/              # API endpoints
│   │   ├── models.py        # Model management
│   │   ├── training.py      # Training endpoints
│   │   ├── predictions.py   # Prediction endpoints
│   │   ├── datasets.py      # Dataset endpoints
│   │   └── visualizations.py # Visualization endpoints
│   └── middleware/          # Custom middleware
├── requirements.txt         # Python dependencies
└── start_server.sh         # Server startup script
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Configure environment (optional):**
```bash
# Create .env file if needed
echo "API_HOST=0.0.0.0" > .env
echo "API_PORT=8000" >> .env
```

## Running the Server

### Development Mode

```bash
cd backend
./start_server.sh
```

Or manually:
```bash
cd backend/api
uvicorn main:app --reload --port 8000
```

### Production Mode

```bash
cd backend/api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

Base URL: `http://localhost:8000`

### Health Check
- `GET /` - Server status

### Models
- `GET /api/models/` - List all trained models
- `GET /api/models/configs` - Get model configurations
- `GET /api/models/compare` - Compare all models
- `GET /api/models/{model_type}` - Get specific model info
- `GET /api/models/{model_type}/details` - Get detailed model info
- `DELETE /api/models/{model_type}` - Delete a model

### Training
- `POST /api/training/start` - Start training job
- `GET /api/training/status/{job_id}` - Get training status
- `GET /api/training/jobs` - List all training jobs
- `DELETE /api/training/jobs/{job_id}` - Delete training job

### Predictions
- `POST /api/predictions/predict` - Make single prediction
- `POST /api/predictions/batch-predict` - Batch predictions
- `GET /api/predictions/features` - Get feature schema

### Datasets
- `GET /api/datasets/info` - Get dataset statistics
- `GET /api/datasets/sample` - Get sample data
- `GET /api/datasets/column/{column}` - Get column info
- `GET /api/datasets/correlation` - Get correlation matrix

### Visualizations
- `GET /api/visualizations/{model_type}/confusion-matrix` - Confusion matrix image
- `GET /api/visualizations/{model_type}/feature-importance` - Feature importance image
- `GET /api/visualizations/{model_type}/shap-summary` - SHAP summary image
- `GET /api/visualizations/{model_type}/status` - Visualization status

## API Documentation

Once the server is running, visit:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Configuration

The backend expects the following directory structure in the parent directory:

```
sri-lanka-employment-predictor/
├── backend/              # This backend project
├── data/                 # Dataset files
├── models/               # Trained model files
├── visualizations/       # Generated plots
├── modules/              # Shared Python modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
└── config.py            # Configuration file
```

## Environment Variables

Create a `.env` file in the backend directory (optional):

```env
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

## CORS Configuration

The API is configured to accept requests from:
- `http://localhost:3000` (React dev server)
- `http://localhost:5173` (Vite dev server)

To add more origins, edit `backend/api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://your-frontend-domain.com"],
    ...
)
```

## Development

### Adding New Endpoints

1. Create a new route file in `backend/api/routes/`
2. Define your endpoints using FastAPI decorators
3. Import and include the router in `main.py`

Example:
```python
# backend/api/routes/my_feature.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/my-endpoint")
async def my_endpoint():
    return {"message": "Hello"}
```

```python
# backend/api/main.py
from routes import my_feature

app.include_router(my_feature.router, prefix="/api/my-feature", tags=["My Feature"])
```

### Background Tasks

Training uses FastAPI's `BackgroundTasks` for async processing:

```python
from fastapi import BackgroundTasks

@router.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(train_model_task, job_id)
    return {"job_id": job_id}
```

## Testing

### Manual Testing

```bash
# Test health endpoint
curl http://localhost:8000/

# List models
curl http://localhost:8000/api/models/

# Get dataset info
curl http://localhost:8000/api/datasets/info

# Make prediction
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

## Deployment

**Important:** GitHub Pages only hosts static websites. For the backend API, use one of these services:

### Option 1: Railway (Recommended - Free Tier Available)

1. **Create account:** https://railway.app

2. **Deploy:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
cd backend
railway init

# Deploy
railway up
```

3. **Configure:**
- Add environment variables in Railway dashboard
- Copy your deployment URL (e.g., `https://your-app.railway.app`)

### Option 2: Render (Free Tier)

1. **Create account:** https://render.com

2. **Create Web Service:**
- Connect GitHub repository
- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables:**
```env
PYTHON_VERSION=3.10
```

### Option 3: Heroku

1. **Install Heroku CLI:**
```bash
brew install heroku/brew/heroku
```

2. **Create app:**
```bash
cd backend
heroku create your-app-name
```

3. **Create Procfile:**
```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

4. **Deploy:**
```bash
git add .
git commit -m "Deploy backend"
git push heroku main
```

### Option 4: AWS Lambda + API Gateway

Use AWS SAM or Serverless Framework:

```yaml
# serverless.yml
service: employment-predictor-api

provider:
  name: aws
  runtime: python3.10

functions:
  api:
    handler: api.main.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
```

Deploy:
```bash
serverless deploy
```

### Option 5: Google Cloud Run

```bash
# Build container
docker build -t employment-predictor-api .

# Push to Google Container Registry
docker tag employment-predictor-api gcr.io/your-project/employment-predictor-api
docker push gcr.io/your-project/employment-predictor-api

# Deploy
gcloud run deploy employment-predictor-api \
  --image gcr.io/your-project/employment-predictor-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Docker Deployment

Create `Dockerfile` in backend directory:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t employment-predictor-backend .
docker run -p 8000:8000 employment-predictor-backend
```

### After Backend Deployment

Once deployed, note your backend URL (e.g., `https://your-app.railway.app`) and configure the frontend to use it.

Update frontend `.env`:
```env
VITE_API_URL=https://your-app.railway.app
```

### Docker (Recommended)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t employment-predictor-api .
docker run -p 8000:8000 employment-predictor-api
```

### Cloud Deployment

#### AWS Elastic Beanstalk
```bash
eb init -p python-3.10 employment-predictor-api
eb create employment-predictor-env
eb deploy
```

#### Google Cloud Run
```bash
gcloud run deploy employment-predictor-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Heroku
```bash
heroku create employment-predictor-api
git push heroku main
```

## Troubleshooting

### Port Already in Use
```bash
# Find process on port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)
```

### Import Errors
Make sure you're running from the project root:
```bash
cd /path/to/sri-lanka-employment-predictor
python -m uvicorn backend.api.main:app --reload
```

### Missing Modules
```bash
cd backend
pip install -r requirements.txt
```

### CORS Errors
Add your frontend URL to the CORS middleware in `backend/api/main.py`

## Performance

- Single worker handles ~1000 req/s for predictions
- Training jobs run in background without blocking
- Model loading is cached for better performance
- Images served directly from disk

## Security Considerations

- No authentication implemented (add JWT/OAuth for production)
- CORS configured for development (restrict in production)
- Input validation using Pydantic models
- File upload validation needed for production

## License

See main project LICENSE file.

## Support

For issues and questions, see the main project repository.

---

**Server Status:** Ready for development
**API Version:** 1.0
**Last Updated:** January 15, 2026
