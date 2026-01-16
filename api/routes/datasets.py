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
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'column_names': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
        }
        
        # Statistics - handle NaN values and convert numpy types
        stats_df = df.describe()
        numeric_stats = {}
        for col in stats_df.columns:
            numeric_stats[col] = {}
            for idx in stats_df.index:
                val = stats_df.loc[idx, col]
                # Convert to native Python type and handle NaN
                if pd.isna(val):
                    numeric_stats[col][idx] = None
                else:
                    numeric_stats[col][idx] = float(val)
        
        # Class distribution if Employment_Status_Encoded exists
        if 'Employment_Status_Encoded' in df.columns:
            class_dist = df['Employment_Status_Encoded'].value_counts().to_dict()
            info['class_distribution'] = {
                'Unemployed': int(class_dist.get(0, 0)),
                'Employed': int(class_dist.get(1, 0))
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
        
        # Replace NaN with None for JSON serialization
        sample = sample.replace({float('nan'): None})
        
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
        
        # Add statistics for numeric columns - handle NaN
        if pd.api.types.is_numeric_dtype(col):
            mean_val = col.mean()
            std_val = col.std()
            min_val = col.min()
            max_val = col.max()
            median_val = col.median()
            
            info.update({
                'mean': None if pd.isna(mean_val) else float(mean_val),
                'std': None if pd.isna(std_val) else float(std_val),
                'min': None if pd.isna(min_val) else float(min_val),
                'max': None if pd.isna(max_val) else float(max_val),
                'median': None if pd.isna(median_val) else float(median_val)
            })
        
        # Add value counts for categorical or low-cardinality columns
        if col.nunique() < 20:
            value_counts = col.value_counts().to_dict()
            # Convert keys and values to JSON-safe types
            info['value_counts'] = {str(k): int(v) for k, v in value_counts.items()}
        
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
        
        # Replace NaN with None for JSON serialization
        corr_matrix = corr_matrix.replace({float('nan'): None})
        
        return {
            'columns': corr_matrix.columns.tolist(),
            'matrix': corr_matrix.values.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
