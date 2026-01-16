"""
FastAPI Backend for Sri Lanka Employment Predictor
Main application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import models, training, predictions, datasets, visualizations

# Create FastAPI app
app = FastAPI(
    title="Sri Lanka Employment Predictor API",
    description="RESTful API for employment prediction using multiple ML models",
    version="1.0.0"
)

# CORS middleware - Allow specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://e10indika.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for visualizations
visualizations_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "visualizations")
if os.path.exists(visualizations_path):
    app.mount("/visualizations", StaticFiles(directory=visualizations_path), name="visualizations")

# Include routers
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(visualizations.router, prefix="/api/visualizations", tags=["visualizations"])

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Sri Lanka Employment Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
