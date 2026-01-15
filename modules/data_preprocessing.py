"""
Data preprocessing module for Sri Lanka Employment Predictor.
Handles data loading, cleaning, feature engineering, and splitting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN, 
    TEST_SIZE, RANDOM_STATE, SCALER_PATH, SAMPLE_PATH,
    COLUMNS_TO_EXCLUDE, save_feature_columns
)


class DataPreprocessor:
    """Handles all data preprocessing operations for employment data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []  # Will be set after preprocessing
        self.target_column = TARGET_COLUMN
    
    def load_data(self, file_path=None):
        """Sri Lanka employment dataset from CSV.
        
        Args:
            file_path: Path to CSV file. If None, uses default path.
        
        Returns:
            pandas DataFrame
        """
        if file_path is None:
            file_path = RAW_DATA_PATH
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load the CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Could not parse CSV file: {e}")
        
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values.
        
        Args:
            df: pandas DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        if df_clean.isnull().sum().sum() > 0:
            print("Missing values detected. Filling with median for numeric columns...")
            for col in df_clean.columns:
                if df_clean[col].dtype in ['float64', 'int64']:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)
        
        # Remove duplicates
        original_size = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < original_size:
            print(f"Removed {original_size - len(df_clean)} duplicate rows")
        
        return df_clean
    
    def engineer_features(self, df):
        """
        Create employment-specific features.
        
        Args:
            df: pandas DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        df_eng = df.copy()
        
        print("Engineering features for employment prediction...")
        
        # 1. Create language profile
        def create_language_profile(row):
            languages = []
            if row['SIN'] == 1:
                languages.append('SIN')
            if row['ENG'] == 1:
                languages.append('ENG')
            if row['TAMIL'] == 1:
                languages.append('TAMIL')
            
            if not languages:
                return 'None'
            else:
                return '+'.join(sorted(languages))
        
        df_eng['Language_Profile'] = df_eng.apply(create_language_profile, axis=1)
        df_eng['Language_Profile_Encoded'] = self.label_encoder.fit_transform(df_eng['Language_Profile'])
        df_eng = df_eng.drop(columns=['SIN', 'ENG', 'TAMIL'])
        
        # 2. Handle Employment columns
        if 'Employment' in df_eng.columns and 'Employment_2' in df_eng.columns:
            if df_eng['Employment'].equals(df_eng['Employment_2']):
                print("'Employment' and 'Employment_2' columns are identical. Dropping 'Employment_2'.")
                df_eng = df_eng.drop(columns=['Employment_2'])
                df_eng = df_eng.rename(columns={'Employment': 'Employment_Status_Encoded'})
            else:
                print("'Employment' and 'Employment_2' columns are different. Creating combined 'Employment_Status'.")
                
                def get_employment_status(row):
                    if row['Employment'] == 0 and row['Employment_2'] == 0:
                        return 'Unemployed'
                    else:
                        return 'Employed'
                
                df_eng['Employment_Status_Categorical'] = df_eng.apply(get_employment_status, axis=1)
                
                employment_status_mapping = {
                    'Unemployed': 0,
                    'Employed': 1
                }
                df_eng['Employment_Status_Encoded'] = df_eng['Employment_Status_Categorical'].map(employment_status_mapping)
                
                df_eng = df_eng.drop(columns=['Employment', 'Employment_2'])
        
        # 3. Disability features
        disability_columns = [
            'Eye Disability', 'Hearing Disability', 'Walking Disability',
            'Remembering Disability', 'Self Care Disability', 'Communicating Disability'
        ]
        
        if all(col in df_eng.columns for col in disability_columns):
            # Disability count
            df_eng['Disability_Count'] = (df_eng[disability_columns] > 1).sum(axis=1)
            
            # Disability severity score
            df_eng['Disability_Severity_Score'] = df_eng[disability_columns].sum(axis=1)
            
            # Max disability severity
            df_eng['Max_Disability_Severity'] = df_eng[disability_columns].max(axis=1)
            
            # Disability category
            def get_disability_category(score):
                if score == 6:
                    return 'None'
                elif 7 <= score <= 9:
                    return 'Mild'
                elif 10 <= score <= 15:
                    return 'Moderate'
                else:
                    return 'Severe'
            
            df_eng['Disability_Category'] = df_eng['Disability_Severity_Score'].apply(get_disability_category)
            
            disability_category_mapping = {
                'None': 0,
                'Mild': 1,
                'Moderate': 2,
                'Severe': 3
            }
            df_eng['Disability_Category_Encoded'] = df_eng['Disability_Category'].map(disability_category_mapping)
            
            # Drop original disability columns and intermediate features
            columns_to_drop = disability_columns + [
                'Disability_Count', 'Disability_Severity_Score', 'Max_Disability_Severity'
            ]
            df_eng = df_eng.drop(columns=[col for col in columns_to_drop if col in df_eng.columns])
        
        print(f"Feature engineering complete. Shape: {df_eng.shape}")
        
        return df_eng
    
    def split_data(self, df, test_size=None, random_state=None):
        """
        Split data into train and test sets.
        
        Args:
            df: pandas DataFrame
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if test_size is None:
            test_size = TEST_SIZE
        if random_state is None:
            random_state = RANDOM_STATE
        
        # Validate target exists
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Determine feature columns by excluding target and other non-feature columns
        columns_to_exclude_from_features = [self.target_column] + [
            col for col in COLUMNS_TO_EXCLUDE if col in df.columns
        ]
        
        self.feature_columns = [col for col in df.columns if col not in columns_to_exclude_from_features]
        
        print(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns}")
        
        # Save feature columns for later use
        save_feature_columns(self.feature_columns)
        
        # Generate and save feature info for the UI
        self._save_feature_info(df)
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Class distribution in train set:")
        print(f"  Unemployed (0): {(y_train == 0).sum()}")
        print(f"  Employed (1): {(y_train == 1).sum()}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test, save_scaler=True):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            save_scaler: Whether to save the fitted scaler
        
        Returns:
            X_train_scaled, X_test_scaled
        """
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if save_scaler:
            joblib.dump(self.scaler, SCALER_PATH)
            print(f"Scaler saved to {SCALER_PATH}")
        
        return X_train_scaled, X_test_scaled
    
    def prepare_data_pipeline(self, file_path=None):
        """
        Complete data preprocessing pipeline.
        
        Args:
            file_path: Path to raw data file
        
        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test, raw_df
        """
        print("Starting data preprocessing pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_eng = self.engineer_features(df_clean)
        
        # Save processed data
        df_eng.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Processed data saved to {PROCESSED_DATA_PATH}")
        
        # Save sample for UI
        sample_df = df_eng.head(100)
        sample_df.to_csv(SAMPLE_PATH, index=False)
        print(f"Sample dataset saved to {SAMPLE_PATH}")
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df_eng)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("Data preprocessing complete!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df_eng

    def _save_feature_info(self, df):
        """
        Save feature information including ranges and options for UI.
        
        Args:
            df: DataFrame with engineered features
        """
        import json
        
        # Feature descriptions based on domain knowledge
        feature_descriptions = {
            'DISTRICT': {'label': 'District', 'type': 'select', 'options': {
                11: 'Colombo', 12: 'Gampaha', 13: 'Kaluthara',
                21: 'Kandy', 22: 'Matale', 23: 'Nuwara Eliya',
                31: 'Galle', 32: 'Mathara', 33: 'Hambanthota',
                41: 'Jaffna', 42: 'Mannar', 43: 'Vavuniya', 44: 'Mulathivu', 45: 'Kilinochchi',
                51: 'Batticaloa', 52: 'Ampara', 53: 'Trincomalee',
                61: 'Kurunegala', 62: 'Puttalam',
                71: 'Anuradhapura', 72: 'Polonnaruwa',
                81: 'Badulla', 82: 'Moneragala',
                91: 'Rathnapura', 92: 'Kegalle'
            }},
            'SEX': {'label': 'Sex', 'type': 'select', 'options': {1: 'Male', 2: 'Female'}},
            'AGE': {'label': 'Age', 'type': 'number'},
            'MARITAL': {'label': 'Marital Status', 'type': 'select', 'options': {
                1: 'Never Married', 2: 'Married', 3: 'Widowed', 5: 'Divorced/Separated'
            }},
            'EDU': {'label': 'Education Level', 'type': 'select', 'options': {
                0: 'Studying/Studied Grade 1',
                1: 'Passed Grade 1',
                2: 'Passed Grade 2',
                3: 'Passed Grade 3',
                4: 'Passed Grade 4',
                5: 'Passed Grade 5',
                6: 'Passed Grade 6',
                7: 'Passed Grade 7',
                8: 'Passed Grade 8',
                9: 'Passed Grade 9',
                10: 'Passed Grade 10',
                11: 'Passed G.C.E (O/L) / N.C.E',
                12: 'Passed Grade 12',
                13: 'Passed G.C.E (A/L) / H.N.C.E',
                14: 'Passed G.A.Q. / G.S.Q',
                15: 'Degree',
                16: 'Postgraduate Degree / Diploma',
                17: 'Special Educational Institutions',
                18: 'Post Graduate - M',
                19: 'Post Graduate - PhD'
            }},
            'DEGREE': {'label': 'Degree Type', 'type': 'select', 'options': {
                0: 'No Degree', 1: 'Diploma', 2: 'Bachelor', 6: 'Master', 9: 'PhD', 10: 'Other'
            }},
            'Language_Profile_Encoded': {'label': 'Language Profile', 'type': 'select', 'options': {
                0: 'ENG', 1: 'ENG+SIN', 2: 'ENG+SIN+TAMIL', 3: 'ENG+TAMIL', 
                4: 'None', 5: 'SIN', 6: 'SIN+TAMIL', 7: 'TAMIL'
            }},
            'Disability_Category_Encoded': {'label': 'Disability Category', 'type': 'select', 'options': {
                0: 'None', 1: 'Mild', 2: 'Moderate'
            }}
        }
        
        feature_info = {}
        for col in self.feature_columns:
            if col in df.columns:
                unique_vals = sorted(df[col].unique())
                feature_info[col] = {
                    'min': int(df[col].min()),
                    'max': int(df[col].max()),
                    'unique_count': int(df[col].nunique()),
                    'unique_values': [int(v) for v in unique_vals],
                    **feature_descriptions.get(col, {'label': col, 'type': 'number'})
                }
        
        # Save to file
        feature_info_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feature_info.json')
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"Feature info saved to {feature_info_path}")


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, df = preprocessor.prepare_data_pipeline()
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
