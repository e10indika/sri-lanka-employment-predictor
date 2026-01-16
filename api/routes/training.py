"""
Training API Routes
Endpoints for training new models
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from modules.data_preprocessing import DataPreprocessor
from modules.model_training import ModelTrainer
from modules.model_evaluation import ModelEvaluator
from config import RAW_DATA_PATH, MODEL_CONFIGS, MODEL_DIR, get_model_viz_paths
import glob

router = APIRouter()

# Store training status
training_jobs = {}

class TrainingRequest(BaseModel):
    model_type: str
    perform_cv: bool = True
    cv_folds: int = 5
    perform_tuning: bool = False
    
class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class TrainingStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: int
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

def train_model_task(job_id: str, request: TrainingRequest):
    """Background task for model training"""
    try:
        training_jobs[job_id]['status'] = 'running'
        training_jobs[job_id]['progress'] = 5
        training_jobs[job_id]['message'] = 'Deleting old model...'
        
        # Delete old model directory if it exists
        model_dir = os.path.join(MODEL_DIR, request.model_type)
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
            training_jobs[job_id]['message'] = f'Old {request.model_type} model deleted'
        
        # Delete old visualizations
        viz_paths = get_model_viz_paths(request.model_type)
        for path in viz_paths.values():
            if os.path.exists(path):
                os.remove(path)
        
        training_jobs[job_id]['progress'] = 10
        training_jobs[job_id]['message'] = 'Loading data...'
        
        # Data preprocessing
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data_pipeline(RAW_DATA_PATH)
        
        training_jobs[job_id]['progress'] = 30
        training_jobs[job_id]['message'] = f'Training {request.model_type}...'
        
        # Training
        trainer = ModelTrainer(model_type=request.model_type)
        
        # Cross-validation
        cv_scores = None
        if request.perform_cv:
            cv_scores = trainer.cross_validate(X_train, y_train, cv=request.cv_folds)
            training_jobs[job_id]['progress'] = 50
        
        # Train model
        if request.perform_tuning:
            # Simplified param grid for API
            param_grid = {}
            if request.model_type == 'xgboost':
                param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'n_estimators': [100, 200]
                }
            elif request.model_type in ['random_forest', 'gradient_boosting']:
                param_grid = {
                    'max_depth': [5, 10, 15],
                    'n_estimators': [100, 200]
                }
            
            if param_grid:
                best_params, model = trainer.hyperparameter_tuning(X_train, y_train, param_grid=param_grid, cv=3)
            else:
                model = trainer.train_model(X_train, y_train, verbose=False)
        else:
            model = trainer.train_model(X_train, y_train, verbose=False)
        
        training_jobs[job_id]['progress'] = 70
        training_jobs[job_id]['message'] = 'Evaluating model...'
        
        # Evaluation
        evaluator = ModelEvaluator(model, X_test, y_test, model_type=request.model_type)
        feature_names = preprocessor.feature_columns if hasattr(preprocessor, 'feature_columns') else None
        metrics = evaluator.evaluate_model(feature_names=feature_names, generate_plots=True)
        
        training_jobs[job_id]['progress'] = 90
        training_jobs[job_id]['message'] = 'Saving model and SHAP explainer...'
        
        # Save model with SHAP explainer
        trainer.save_model(X_train=X_train)
        
        training_jobs[job_id]['progress'] = 100
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['message'] = 'Training completed successfully'
        training_jobs[job_id]['results'] = {
            'metrics': metrics,
            'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
            'model_type': request.model_type,
            'model_name': trainer.model_name
        }
        
    except Exception as e:
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)
        training_jobs[job_id]['message'] = f'Training failed: {str(e)}'

@router.post("/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new model training job"""
    try:
        # Validate model type
        if request.model_type not in MODEL_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")
        
        # Create job ID
        job_id = f"training_{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize job status
        training_jobs[job_id] = {
            'job_id': job_id,
            'status': 'pending',
            'progress': 0,
            'message': 'Training job queued',
            'results': None,
            'error': None
        }
        
        # Start background task
        background_tasks.add_task(train_model_task, job_id, request)
        
        return TrainingResponse(
            job_id=job_id,
            status='pending',
            message='Training job started'
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get status of a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    return TrainingStatus(**training_jobs[job_id])

@router.get("/jobs", response_model=List[TrainingStatus])
async def list_training_jobs():
    """List all training jobs"""
    return [TrainingStatus(**job) for job in training_jobs.values()]

@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """Delete a training job from history"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    del training_jobs[job_id]
    return {"message": f"Training job {job_id} deleted"}
