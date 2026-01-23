import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import FraudDataPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class XGBoostTuner:
    """Hyperparameter tuning for XGBoost using Optuna"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = None
        self.X_processed = None
        self.y = None
        self.best_params = None
        
    def load_and_preprocess(self):
        """Load and preprocess data for tuning"""
        data_path = Path(self.config['data']['processed_path']) / 'engineered_data.parquet'
        
        # Load the data
        try:
            df = pd.read_parquet(data_path)
        except Exception as e:
            logger.error(f"Error loading parquet: {e}")
            # Try joblib format as fallback
            try:
                df = joblib.load(data_path)
            except Exception as e2:
                logger.error(f"Error loading joblib: {e2}")
                raise ValueError(f"Cannot load data from {data_path}")
        
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        
        # Preprocess
        self.preprocessor = FraudDataPreprocessor(self.config_path)
        self.X_processed = self.preprocessor.fit_transform(X, y)
        self.y = y.values
        
        logger.info(f"Data loaded for tuning. Shape: {self.X_processed.shape}")
        logger.info(f"Class distribution: {np.bincount(self.y)}")
    
    def objective(self, trial):
        """Optuna objective function"""
        
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': self.config['model']['random_state'],
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        # Use stratified k-fold cross-validation
        cv = StratifiedKFold(
            n_splits=self.config['model']['cv_folds'],
            shuffle=True,
            random_state=self.config['model']['random_state']
        )
        
        scores = []
        
        for train_idx, val_idx in cv.split(self.X_processed, self.y):
            X_train, X_val = self.X_processed[train_idx], self.X_processed[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model = xgb.XGBClassifier(**param)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)
        
        return np.mean(scores)
    
    def tune(self, n_trials: int = 50):
        """Run hyperparameter tuning"""
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name='xgboost_fraud_detection',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        study.optimize(self.objective, n_trials=n_trials, n_jobs=-1)
        
        self.best_params = study.best_params
        logger.info(f"Best AUC: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Save study
        models_path = Path(self.config['data']['models_path'])
        models_path.mkdir(parents=True, exist_ok=True)
        
        study_path = models_path / 'optuna_study.joblib'
        joblib.dump(study, study_path)
        
        return study
    
    def train_best_model(self):
        """Train final model with best parameters"""
        if self.best_params is None:
            raise ValueError("No best parameters found. Run tune() first.")
        
        logger.info("Training final model with best parameters...")
        
        # Add fixed parameters
        final_params = self.best_params.copy()
        final_params.update({
            'random_state': self.config['model']['random_state'],
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        })
        
        # Train final model on all data
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(self.X_processed, self.y)
        
        # Save final model
        models_path = Path(self.config['data']['models_path'])
        model_path = models_path / 'tuned_xgboost_model.joblib'
        joblib.dump(final_model, model_path)
        
        logger.info(f"Final model saved to {model_path}")
        
        return final_model


def main():
    """Main tuning pipeline"""
    logger.info("Starting hyperparameter tuning pipeline...")
    
    tuner = XGBoostTuner()
    
    # Load and preprocess data
    tuner.load_and_preprocess()
    
    # Run tuning - start with fewer trials for testing
    study = tuner.tune(n_trials=10)  # Start with 10, then increase to 50
    
    # Train final model
    final_model = tuner.train_best_model()
    
    logger.info("Hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()