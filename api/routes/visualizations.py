"""
Visualizations API Routes
Endpoints for accessing model visualizations
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import get_model_viz_paths, VIZ_DIR

router = APIRouter()

@router.get("/{model_type}/confusion-matrix")
async def get_confusion_matrix(model_type: str):
    """Get confusion matrix image for a model"""
    try:
        viz_paths = get_model_viz_paths(model_type)
        path = viz_paths['confusion_matrix']
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Confusion matrix not found for {model_type}")
        
        return FileResponse(path, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_type}/feature-importance")
async def get_feature_importance(model_type: str):
    """Get feature importance plot for a model"""
    try:
        viz_paths = get_model_viz_paths(model_type)
        path = viz_paths['feature_importance']
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Feature importance plot not found for {model_type}")
        
        return FileResponse(path, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_type}/shap-summary")
async def get_shap_summary(model_type: str):
    """Get SHAP summary plot for a model"""
    try:
        viz_paths = get_model_viz_paths(model_type)
        path = viz_paths['shap_summary']
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"SHAP summary plot not found for {model_type}")
        
        return FileResponse(path, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_type}/lime-explanation")
async def get_lime_explanation(model_type: str):
    """Get LIME explanation plot for a model"""
    try:
        path = os.path.join(VIZ_DIR, f'lime_explanation_{model_type}.png')
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"LIME explanation plot not found for {model_type}")
        
        return FileResponse(path, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_type}/partial-dependence")
async def get_partial_dependence(model_type: str):
    """Get Partial Dependence Plot for a model"""
    try:
        path = os.path.join(VIZ_DIR, f'partial_dependence_{model_type}.png')
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Partial Dependence Plot not found for {model_type}")
        
        return FileResponse(path, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_type}/status")
async def get_visualization_status(model_type: str):
    """Check which visualizations exist for a model"""
    try:
        viz_paths = get_model_viz_paths(model_type)
        
        # Check all 5 visualization types
        lime_path = os.path.join(VIZ_DIR, f'lime_explanation_{model_type}.png')
        pdp_path = os.path.join(VIZ_DIR, f'partial_dependence_{model_type}.png')
        
        status = {
            'model_type': model_type,
            'visualizations': {
                'confusion_matrix': os.path.exists(viz_paths['confusion_matrix']),
                'feature_importance': os.path.exists(viz_paths['feature_importance']),
                'shap_summary': os.path.exists(viz_paths['shap_summary']),
                'lime_explanation': os.path.exists(lime_path),
                'partial_dependence': os.path.exists(pdp_path)
            },
            'urls': {
                'confusion_matrix': f'/api/visualizations/{model_type}/confusion-matrix',
                'feature_importance': f'/api/visualizations/{model_type}/feature-importance',
                'shap_summary': f'/api/visualizations/{model_type}/shap-summary',
                'lime_explanation': f'/api/visualizations/{model_type}/lime-explanation',
                'partial_dependence': f'/api/visualizations/{model_type}/partial-dependence'
            }
        }
        
        status['all_available'] = all(status['visualizations'].values())
        status['available_count'] = sum(status['visualizations'].values())
        status['total_count'] = len(status['visualizations'])
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
