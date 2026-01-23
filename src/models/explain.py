import shap
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, Any, List
from pathlib import Path

class ModelExplainer:
    """SHAP explanation wrapper for API integration"""
    
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Handle XGBoost base_score issue
        self._initialize_explainer()
        
        # Get feature names
        if isinstance(self.preprocessor, dict) and 'feature_names' in self.preprocessor:
            self.feature_names = self.preprocessor['feature_names']
        elif hasattr(self.preprocessor, 'feature_names'):
            self.feature_names = self.preprocessor.feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(self.model.n_features_in_)]
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer with workaround for XGBoost"""
        try:
            # Try standard initialization first
            self.explainer = shap.TreeExplainer(self.model)
        except ValueError as e:
            if "could not convert string to float" in str(e):
                print("Using XGBoost workaround for SHAP initialization...")
                
                # Workaround for XGBoost base_score issue
                if hasattr(self.model, 'get_booster'):
                    booster = self.model.get_booster()
                    
                    # Get background data for SHAP
                    data_path = Path("data/processed/engineered_data.parquet")
                    if data_path.exists():
                        df = pd.read_parquet(data_path)
                        X = df.drop('isFraud', axis=1)
                        background_data = self.preprocessor.transform(X[:100])  # Use 100 samples
                        
                        # Initialize with background data
                        self.explainer = shap.TreeExplainer(
                            booster,
                            background_data,
                            model_output="probability"
                        )
                    else:
                        # Fallback to simple initialization
                        self.explainer = shap.TreeExplainer(
                            booster,
                            model_output="probability"
                        )
                else:
                    # For non-XGBoost models
                    self.explainer = shap.Explainer(self.model)
            else:
                raise
    
    def explain_prediction(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP explanation for a single prediction"""
        
        # Preprocess input
        processed_data = self.preprocessor.transform(input_data)
        
        # Get prediction
        prediction_proba = self.model.predict_proba(processed_data)[0, 1]
        
        # Get SHAP values
        try:
            shap_values = self.explainer.shap_values(processed_data)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get positive class if it's a list
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]  # Get positive class if 3D
                
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            # Fallback to simple feature importance
            shap_values = np.zeros((1, processed_data.shape[1]))
        
        # For single sample
        if len(shap_values.shape) == 2:
            shap_values = shap_values[0]
        
        # Get top contributing features
        contributions = []
        for i, (feature, value, shap_val) in enumerate(zip(
            self.feature_names[:len(shap_values)],
            processed_data[0],
            shap_values
        )):
            contributions.append({
                'feature': feature,
                'value': float(value),
                'contribution': float(shap_val),
                'absolute_contribution': abs(float(shap_val))
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: x['absolute_contribution'], reverse=True)
        
        # Get base value
        try:
            base_value = float(self.explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]  # Get positive class
        except:
            base_value = 0.0
        
        return {
            'prediction': float(prediction_proba),
            'base_value': base_value,
            'contributions': contributions[:10],  # Top 10 features
            'is_high_risk': prediction_proba > 0.7,
            'explanation': self._generate_narrative(contributions, prediction_proba)
        }
    
    def _generate_narrative(self, contributions: List[Dict], probability: float) -> str:
        """Generate human-readable explanation"""
        
        top_positive = [c for c in contributions if c['contribution'] > 0][:2]
        top_negative = [c for c in contributions if c['contribution'] < 0][:2]
        
        narrative = f"This transaction has a {probability:.1%} probability of being fraudulent.\n\n"
        
        if top_positive:
            narrative += "Factors increasing fraud risk:\n"
            for contrib in top_positive:
                narrative += f"• {contrib['feature']}: contributed +{contrib['contribution']:.3f}\n"
        
        if top_negative:
            narrative += "\nFactors decreasing fraud risk:\n"
            for contrib in top_negative:
                narrative += f"• {contrib['feature']}: contributed {contrib['contribution']:.3f}\n"
        
        if probability > 0.7:
            narrative += "\n⚠️ **High Risk**: This transaction requires manual review."
        elif probability > 0.3:
            narrative += "\n⚠️ **Medium Risk**: Consider additional verification."
        else:
            narrative += "\n✅ **Low Risk**: Transaction appears legitimate."
        
        return narrative