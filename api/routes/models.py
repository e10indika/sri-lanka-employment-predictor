"""
Models API Routes
Endpoints for managing and querying trained models
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import get_available_models, get_model_viz_paths, MODEL_CONFIGS
import joblib

router = APIRouter()

@router.get("/", response_model=List[Dict[str, Any]])
async def list_models():
    """Get list of all available trained models"""
    try:
        models = get_available_models()
        
        # Add visualization status
        for model in models:
            viz_paths = model.get('visualizations', {})
            model['has_visualizations'] = all(
                os.path.exists(path) for path in viz_paths.values()
            ) if viz_paths else False
        
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/configs")
async def get_model_configs():
    """Get available model configurations"""
    return MODEL_CONFIGS

@router.get("/compare")
async def compare_models():
    """Get comparison data for all trained models"""
    try:
        models = get_available_models()
        
        if not models:
            return {
                "models": [],
                "comparison": {},
                "best_model": None,
                "message": "No models found. Train some models first."
            }
        
        comparison_data = []
        models_without_metrics = []
        
        for model_info in models:
            # Load model to get full details
            try:
                model_data = joblib.load(model_info['path'])
                
                if isinstance(model_data, dict):
                    metrics = model_data.get('metrics', {})
                    cv_scores = model_data.get('cv_scores', [])
                    training_time = model_data.get('training_time', 0)
                    
                    # Check if this model has metrics
                    if not metrics:
                        models_without_metrics.append(model_info['model_name'])
                        continue
                else:
                    models_without_metrics.append(model_info['model_name'])
                    continue
                
                comparison_data.append({
                    'model_type': model_info['model_type'],
                    'model_name': model_info['model_name'],
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'cv_mean': float(sum(cv_scores) / len(cv_scores)) if cv_scores else 0,
                    'cv_std': float(max(cv_scores) - min(cv_scores)) if len(cv_scores) > 1 else 0,
                    'cv_scores': [float(s) for s in cv_scores],
                    'training_time': training_time,
                    'has_visualizations': model_info.get('has_visualizations', False)
                })
            except Exception as e:
                print(f"Error loading model {model_info['model_type']}: {e}")
                models_without_metrics.append(model_info['model_name'])
                continue
        
        # Find best model by F1-score
        best_model = None
        if comparison_data:
            best_model = max(comparison_data, key=lambda x: x['f1_score'])['model_type']
        
        result = {
            "models": comparison_data,
            "best_model": best_model,
            "total_models": len(comparison_data)
        }
        
        # Add warning if some models don't have metrics
        if models_without_metrics:
            result["warning"] = f"Some models lack metrics data and were excluded: {', '.join(models_without_metrics)}. Re-train these models using the Train Model page to include them in comparison."
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_type}")
async def get_model_info(model_type: str):
    """Get information about a specific model"""
    try:
        models = get_available_models()
        model = next((m for m in models if m['model_type'] == model_type), None)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
        
        return model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_type}/details")
async def get_model_details(model_type: str):
    """Get detailed information about a model including parameters"""
    try:
        models = get_available_models()
        model = next((m for m in models if m['model_type'] == model_type), None)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
        
        # Load model to get additional details
        model_data = joblib.load(model['path'])
        
        if isinstance(model_data, dict):
            details = {
                **model,
                'params': model_data.get('params', {}),
                'model_class': type(model_data['model']).__name__
            }
        else:
            details = {
                **model,
                'model_class': type(model_data).__name__
            }
        
        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{model_type}")
async def delete_model(model_type: str):
    """Delete a trained model"""
    try:
        models = get_available_models()
        model = next((m for m in models if m['model_type'] == model_type), None)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
        
        # Delete model file
        os.remove(model['path'])
        
        # Delete visualizations
        viz_paths = get_model_viz_paths(model_type)
        for path in viz_paths.values():
            if os.path.exists(path):
                os.remove(path)
        
        return {"message": f"Model {model_type} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
