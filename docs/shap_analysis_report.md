# SHAP Analysis Report - Fraud Detection Model

## Executive Summary
This SHAP analysis reveals the key drivers of fraud predictions in the XGBoost model. The analysis helps understand which features are most important and how they influence individual predictions.

## Key Findings

### Most Important Features
The top 5 features by SHAP importance are:

- **C5**: SHAP importance = 0.498798
- **C14**: SHAP importance = 0.414357
- **C13**: SHAP importance = 0.346834
- **ProductCD_C**: SHAP importance = 0.173315
- **C1**: SHAP importance = 0.155546

### Model Interpretation
- **Expected Value (Base)**: 0.0848
- **Number of Significant Features**: 20
- **Prediction Range**: 0.0013 to 0.9640

## Files Generated
1. `shap_summary.png` - Summary plot of feature importance
2. `shap_importance_bar.png` - Bar chart of top 20 features
3. `shap_dependence.png` - Dependence plot for top feature
4. `shap_force_*.png` - Individual prediction explanations
5. `shap_force_interactive.html` - Interactive force plot
6. `shap_api_data.joblib` - Data for API integration

## Recommendations
1. **Focus monitoring** on the top 5 features for fraud detection
2. **Validate feature engineering** for high-importance features
3. **Consider feature interactions** shown in dependence plots
4. **Use individual explanations** for high-risk cases
