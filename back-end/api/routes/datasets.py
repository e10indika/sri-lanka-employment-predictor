"""
Datasets API Routes
Endpoints for dataset operations
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLE_PATH
from modules.data_preprocessing import DataPreprocessor

router = APIRouter()

@router.post("/preprocess")
async def preprocess_dataset(file: UploadFile = File(None)):
    """
    Run data preprocessing pipeline to generate processed_data.csv
    
    Args:
        file: Optional CSV file upload. If provided, uses this file instead of default raw data.
    """
    try:
        data_path = RAW_DATA_PATH
        temp_file_path = None
        
        # If file is uploaded, save it temporarily and use it
        if file is not None:
            # Validate file type
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="Only CSV files are accepted")
            
            # Save uploaded file temporarily
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            temp_file_path = temp_file.name
            
            # Write uploaded content to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            
            data_path = temp_file_path
        else:
            # Check if default raw data exists
            if not os.path.exists(RAW_DATA_PATH):
                raise HTTPException(status_code=404, detail="Raw dataset not found. Please upload a CSV file.")
        
        # Run preprocessing
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, df = preprocessor.prepare_data_pipeline(data_path)
        
        # Clean up temp file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {
            'status': 'success',
            'message': 'Data preprocessing completed successfully',
            'details': {
                'total_rows': int(len(df)),
                'train_samples': int(len(X_train)),
                'test_samples': int(len(X_test)),
                'features': int(X_train.shape[1]),
                'processed_file': PROCESSED_DATA_PATH,
                'source': 'uploaded_file' if file else 'default_raw_data'
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        # Clean up temp file on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500, 
            detail=f"Preprocessing failed: {str(e)}\n{traceback.format_exc()}"
        )

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

@router.get("/row/{row_number}")
async def get_row_data(row_number: int):
    """Get data for a specific row number with human-readable labels"""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            raise HTTPException(status_code=404, detail="Processed dataset not found")
        
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        # Check if row number is valid (0-indexed)
        if row_number < 0 or row_number >= len(df):
            raise HTTPException(
                status_code=404, 
                detail=f"Row {row_number} not found. Dataset has {len(df)} rows (0-{len(df)-1})"
            )
        
        # Load feature info for label mapping
        feature_info = {}
        feature_info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'feature_info.json')
        if os.path.exists(feature_info_path):
            import json
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
        
        # Get the row
        row = df.iloc[row_number]
        
        # Convert to dictionary and handle different data types
        row_dict = {}
        row_labels = {}  # Store human-readable labels
        
        for k, v in row.to_dict().items():
            # Convert value to appropriate type
            if pd.isna(v):
                row_dict[k] = None
                row_labels[k] = None
            elif isinstance(v, (bool, np.bool_)):
                row_dict[k] = bool(v)
                row_labels[k] = str(v)
            elif isinstance(v, (int, np.integer)):
                row_dict[k] = int(v)
                # Try to get human-readable label from feature_info
                if k in feature_info and 'options' in feature_info[k]:
                    label = feature_info[k]['options'].get(str(int(v)))
                    row_labels[k] = label if label else str(int(v))
                else:
                    row_labels[k] = str(int(v))
            elif isinstance(v, (float, np.floating)):
                row_dict[k] = float(v)
                row_labels[k] = str(float(v))
            else:
                # Keep strings and other types as-is
                row_dict[k] = str(v)
                row_labels[k] = str(v)
        
        return {
            'row_number': row_number,
            'data': row_dict,
            'labels': row_labels,  # Human-readable labels
            'total_rows': int(len(df))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
