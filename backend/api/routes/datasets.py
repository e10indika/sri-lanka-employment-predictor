"""
Datasets API Routes
Endpoints for dataset operations
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLE_PATH

router = APIRouter()

@router.get("/info")
async def get_dataset_info():
    """Get dataset information and statistics"""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            raise HTTPException(status_code=404, detail="Processed dataset not found")
        
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        # Basic info
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Statistics
        numeric_stats = df.describe().to_dict()
        
        # Class distribution if Employment_Status_Encoded exists
        if 'Employment_Status_Encoded' in df.columns:
            class_dist = df['Employment_Status_Encoded'].value_counts().to_dict()
            info['class_distribution'] = {
                'Unemployed': class_dist.get(0, 0),
                'Employed': class_dist.get(1, 0)
            }
        
        return {
            'info': info,
            'statistics': numeric_stats
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sample")
async def get_sample_data(n: int = 10):
    """Get sample rows from dataset"""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            raise HTTPException(status_code=404, detail="Processed dataset not found")
        
        df = pd.read_csv(PROCESSED_DATA_PATH)
        sample = df.head(n)
        
        return {
            'data': sample.to_dict(orient='records'),
            'columns': sample.columns.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/column/{column_name}")
async def get_column_info(column_name: str):
    """Get detailed information about a specific column"""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            raise HTTPException(status_code=404, detail="Processed dataset not found")
        
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        if column_name not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column {column_name} not found")
        
        col = df[column_name]
        
        info = {
            'name': column_name,
            'dtype': str(col.dtype),
            'count': int(col.count()),
            'missing': int(col.isnull().sum()),
            'unique': int(col.nunique())
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(col):
            info.update({
                'mean': float(col.mean()),
                'std': float(col.std()),
                'min': float(col.min()),
                'max': float(col.max()),
                'median': float(col.median())
            })
        
        # Add value counts for categorical or low-cardinality columns
        if col.nunique() < 20:
            info['value_counts'] = col.value_counts().to_dict()
        
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/correlation")
async def get_correlation_matrix():
    """Get correlation matrix for numeric features"""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            raise HTTPException(status_code=404, detail="Processed dataset not found")
        
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        
        if len(numeric_df.columns) == 0:
            raise HTTPException(status_code=400, detail="No numeric columns found")
        
        corr_matrix = numeric_df.corr()
        
        return {
            'columns': corr_matrix.columns.tolist(),
            'matrix': corr_matrix.values.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
