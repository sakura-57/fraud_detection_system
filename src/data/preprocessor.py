import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import yaml
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDataPreprocessor:
    """Production-ready data preprocessing pipeline"""
    
    def __init__(self, config_path: str = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._load_default_config()
        
        self.preprocessor = None
        self.feature_names = None
        
    def _load_default_config(self) -> dict:
        """Load default configuration"""
        return {
            'categorical_cols': ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'DeviceType'],
            'numerical_cols': ['TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5'],
            'engineered_cols': []
        }
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline"""
        
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Get columns from config
        numerical_features = self.config['features']['numerical_cols']
        categorical_features = self.config['features']['categorical_cols']
        engineered_features = self.config['features']['engineered_cols']
        
        all_numerical = numerical_features + engineered_features
        
        # Create column transformer
        preprocessor = ColumnTransformer([
            ('numerical', numerical_pipeline, all_numerical),
            ('categorical', categorical_pipeline, categorical_features)
        ])
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit and transform the data"""
        if self.preprocessor is None:
            self.create_preprocessing_pipeline()
        
        X_processed = self.preprocessor.fit_transform(X, y)
        
        # Get feature names
        self._extract_feature_names(X)
        
        return X_processed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        return self.preprocessor.transform(X)
    
    def _extract_feature_names(self, X: pd.DataFrame):
        """Extract feature names after one-hot encoding"""
        if self.preprocessor is None:
            return
        
        feature_names = []
        
        # Get numerical feature names
        numerical_features = self.config['features']['numerical_cols'] + self.config['features']['engineered_cols']
        feature_names.extend(numerical_features)
        
        # Get categorical feature names
        categorical_features = self.config['features']['categorical_cols']
        categorical_transformer = self.preprocessor.named_transformers_['categorical']
        
        if hasattr(categorical_transformer, 'named_steps'):
            onehot = categorical_transformer.named_steps['onehot']
            if hasattr(onehot, 'get_feature_names_out'):
                cat_feature_names = onehot.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
        
        self.feature_names = feature_names
    
    def save(self, path: str):
        """Save the preprocessor"""
        if self.preprocessor is None:
            raise ValueError("No preprocessor to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'preprocessor': self.preprocessor,
            'config': self.config,
            'feature_names': self.feature_names
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load a saved preprocessor"""
        saved_data = joblib.load(path)
        
        preprocessor = cls()
        preprocessor.preprocessor = saved_data['preprocessor']
        preprocessor.config = saved_data['config']
        preprocessor.feature_names = saved_data['feature_names']
        
        return preprocessor


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer for feature selection"""
    
    def __init__(self, columns: list):
        self.columns = columns
        self.selected_columns = None
    
    def fit(self, X: pd.DataFrame, y=None):
        # Check which columns exist
        self.selected_columns = [col for col in self.columns if col in X.columns]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_columns]