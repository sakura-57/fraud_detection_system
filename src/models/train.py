import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import FraudDataPreprocessor
from src.utils.logger import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns

logger = setup_logger(__name__)

class FraudModelTrainer:
    """Model training and evaluation pipeline"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = None
        self.models = {}
        self.results = {}
        
    def load_and_split_data(self) -> tuple:
        """Load engineered data and split into train/test"""
        data_path = Path(self.config['data']['processed_path']) / 'engineered_data.parquet'
        df = pd.read_parquet(data_path)
        
        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Fraud rate: {df['isFraud'].mean():.4%}")
        
        # Split features and target
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def create_models(self):
        """Initialize models for benchmarking"""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.config['model']['random_state'],
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config['model']['random_state'],
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=self.config['xgboost']['n_estimators'],
                max_depth=self.config['xgboost']['max_depth'],
                learning_rate=self.config['xgboost']['learning_rate'],
                subsample=self.config['xgboost']['subsample'],
                colsample_bytree=self.config['xgboost']['colsample_bytree'],
                random_state=self.config['xgboost']['random_state'],
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.config['model']['random_state'],
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        self.models = models
        return models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        
        # Create and fit preprocessor
        self.preprocessor = FraudDataPreprocessor(self.config_path)
        X_train_processed = self.preprocessor.fit_transform(X_train, y_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        logger.info(f"Processed train shape: {X_train_processed.shape}")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_processed, y_train)
            
            # Predictions
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Store results
            results[name] = {
                'model': model,
                'auc': auc,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred,
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1': report['1']['f1-score']
            }
            
            logger.info(f"{name} - AUC: {auc:.4f}, Precision: {report['1']['precision']:.4f}, "
                       f"Recall: {report['1']['recall']:.4f}")
        
        self.results = results
        return results
    
    def plot_results(self, y_test):
        """Create visualization of model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. AUC Comparison
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc'] for name in model_names]
        
        axes[0, 0].bar(model_names, auc_scores)
        axes[0, 0].set_title('Model Comparison - AUC Score')
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].set_ylim(0.5, 1.0)
        for i, v in enumerate(auc_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # 2. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve
        for name in model_names:
            precision, recall, _ = precision_recall_curve(
                y_test, self.results[name]['y_pred_proba']
            )
            axes[0, 1].plot(recall, precision, label=name, linewidth=2)
        
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion Matrix for best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['auc'])
        best_result = self.results[best_model_name]
        
        cm = confusion_matrix(y_test, best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Feature Importance for tree-based models
        if hasattr(self.results['xgboost']['model'], 'feature_importances_'):
            importances = self.results['xgboost']['model'].feature_importances_
            if self.preprocessor.feature_names:
                feature_names = self.preprocessor.feature_names
                # Limit to top 20
                indices = np.argsort(importances)[-20:]
                
                axes[1, 1].barh(range(len(indices)), importances[indices])
                axes[1, 1].set_yticks(range(len(indices)))
                axes[1, 1].set_yticklabels([feature_names[i] for i in indices])
                axes[1, 1].set_title('Top 20 Feature Importances (XGBoost)')
                axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('docs/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Best model: {best_model_name} with AUC: {best_result['auc']:.4f}")
    
    def save_best_model(self):
        """Save the best performing model and preprocessor"""
        if not self.results:
            raise ValueError("No results to save. Train models first.")
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['auc'])
        best_model = self.results[best_model_name]['model']
        
        # Save model
        models_path = Path(self.config['data']['models_path'])
        models_path.mkdir(parents=True, exist_ok=True)
        
        model_path = models_path / 'best_model.joblib'
        joblib.dump(best_model, model_path)
        
        # Save preprocessor
        preprocessor_path = models_path / 'preprocessor.joblib'
        self.preprocessor.save(preprocessor_path)
        
        # Save results
        results_path = models_path / 'training_results.joblib'
        joblib.dump(self.results, results_path)
        
        logger.info(f"Saved best model ({best_model_name}) to {model_path}")
        logger.info(f"Saved preprocessor to {preprocessor_path}")


def main():
    """Main training pipeline"""
    logger.info("Starting model training pipeline...")
    
    # Initialize trainer
    trainer = FraudModelTrainer()
    
    # Load and split data
    X_train, X_test, y_train, y_test = trainer.load_and_split_data()
    
    # Create models
    trainer.create_models()
    
    # Train and evaluate
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Plot results
    trainer.plot_results(y_test)
    
    # Save best model
    trainer.save_best_model()
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()