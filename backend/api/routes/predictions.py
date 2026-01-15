"""
Predictions API Routes
Endpoints for making predictions
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
import os
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import get_available_models, SCALER_PATH, MODEL_DIR

router = APIRouter()

class PredictionRequest(BaseModel):
    model_type: str
    features: Dict[str, Any]  # Feature name: value pairs
    
class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probabilities: Dict[str, float]
    model_type: str
    model_name: str

@router.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Make a prediction using specified model"""
    try:
        # Get model info
        models = get_available_models()
        model_info = next((m for m in models if m['model_type'] == request.model_type), None)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {request.model_type} not found")
        
        # Load model
        model_data = joblib.load(model_info['path'])
        if isinstance(model_data, dict):
            model = model_data['model']
        else:
            model = model_data
        
        # Load scaler
        scaler = joblib.load(SCALER_PATH)
        
        # Prepare input
        input_df = pd.DataFrame([request.features])
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Map prediction to label
        prediction_label = "Employed" if prediction == 1 else "Unemployed"
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            probabilities={
                "Unemployed": float(prediction_proba[0]),
                "Employed": float(prediction_proba[1])
            },
            model_type=request.model_type,
            model_name=model_info['model_name']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch-predict")
async def batch_predict(model_type: str, features_list: list[Dict[str, Any]]):
    """Make batch predictions"""
    try:
        # Get model info
        models = get_available_models()
        model_info = next((m for m in models if m['model_type'] == model_type), None)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
        
        # Load model and scaler
        model_data = joblib.load(model_info['path'])
        model = model_data['model'] if isinstance(model_data, dict) else model_data
        scaler = joblib.load(SCALER_PATH)
        
        # Prepare inputs
        input_df = pd.DataFrame(features_list)
        input_scaled = scaler.transform(input_df)
        
        # Make predictions
        predictions = model.predict(input_scaled).tolist()
        probabilities = model.predict_proba(input_scaled).tolist()
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append({
                'index': i,
                'prediction': int(pred),
                'prediction_label': "Employed" if pred == 1 else "Unemployed",
                'probabilities': {
                    "Unemployed": float(proba[0]),
                    "Employed": float(proba[1])
                }
            })
        
        return {
            'model_type': model_type,
            'model_name': model_info['model_name'],
            'predictions': results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features")
async def get_feature_schema():
    """Get required features for prediction"""
    try:
        from config import load_feature_columns
        feature_columns = load_feature_columns()
        
        # Feature information with expected ranges
        feature_info = {
            'DISTRICT': {'type': 'integer', 'range': [1, 25], 'description': 'District code (1-25)'},
            'SEX': {'type': 'integer', 'range': [1, 2], 'description': 'Sex (1=Male, 2=Female)'},
            'AGE': {'type': 'integer', 'range': [15, 100], 'description': 'Age in years'},
            'MARITAL': {'type': 'integer', 'range': [1, 6], 'description': 'Marital status code'},
            'EDU': {'type': 'integer', 'range': [1, 20], 'description': 'Education level code'},
            'Language_Profile_Encoded': {'type': 'integer', 'range': [0, 7], 'description': 'Language proficiency'},
            'Disability_Category_Encoded': {'type': 'integer', 'range': [0, 6], 'description': 'Disability category'}
        }
        
        return {
            'features': feature_columns,
            'feature_info': feature_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
