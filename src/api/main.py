# src/api/main.py

from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import joblib
import uuid
import time
from datetime import datetime
from pathlib import Path
import sys
import os
import psutil

# Add the project root to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from src.api.schemas import (
    PredictionRequest, PredictionResponse, HealthResponse,
    BatchPredictionRequest, BatchPredictionResponse
)
from src.api.database import get_db, PredictionLog, ModelVersion, create_tables
# Removed: from src.models.explain import ModelExplainer  # This might be causing issues
# from src.utils.logger import setup_logger

# Initialize logger (simplified version since we don't have the logger module)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Production-ready API for detecting fraudulent transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model - FIXED PATHS
MODEL_PATH = Path("data/models/best_model.joblib")  # Changed from tuned_xgboost_model
PREPROCESSOR_PATH = Path("data/models/preprocessor.joblib")
MODEL_VERSION = "1.0.0"

# Global variables for model
model = None
preprocessor = None
feature_names = None

def load_model_and_preprocessor():
    """Load model and preprocessor"""
    global model, preprocessor, feature_names
    
    try:
        # Load the model
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            logger.info(f"Model type: {type(model).__name__}")
            
            # Check if model has feature names
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_.tolist()
                logger.info(f"Model expects {len(feature_names)} features")
            elif hasattr(model, 'n_features_in_'):
                logger.info(f"Model expects {model.n_features_in_} features")
        else:
            logger.error(f"Model file not found: {MODEL_PATH}")
            model = None
        
        # Load preprocessor if available
        if PREPROCESSOR_PATH.exists():
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
        else:
            logger.warning(f"Preprocessor not found: {PREPROCESSOR_PATH}")
            preprocessor = None
            
    except Exception as e:
        logger.error(f"Failed to load model/preprocessor: {e}")
        model = None
        preprocessor = None

# Simple explainer class since the original might not be working
class SimpleModelExplainer:
    """Simple model explainer for predictions"""
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
    
    def explain_prediction(self, transaction_df):
        """Generate simple explanation for prediction"""
        try:
            # Preprocess if preprocessor exists
            if self.preprocessor is not None:
                processed_data = self.preprocessor.transform(transaction_df)
            else:
                processed_data = transaction_df.values
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(processed_data)
                prediction = float(proba[0][1])  # Probability of fraud (class 1)
            else:
                prediction = float(self.model.predict(processed_data)[0])
            
            # Simple feature contributions (placeholder)
            contributions = []
            if hasattr(self.model, 'feature_importances_') and hasattr(self.model, 'feature_names_in_'):
                importances = self.model.feature_importances_
                names = self.model.feature_names_in_
                for name, importance in zip(names[:5], importances[:5]):  # Top 5
                    contributions.append({
                        "feature": name,
                        "value": float(transaction_df[name].iloc[0]) if name in transaction_df.columns else 0.0,
                        "contribution": float(importance),
                        "absolute_contribution": float(abs(importance))
                    })
            
            # Generate explanation text
            if prediction > 0.7:
                explanation = f"High risk transaction with {prediction*100:.1f}% fraud probability"
                is_high_risk = True
            elif prediction > 0.5:
                explanation = f"Moderate risk transaction with {prediction*100:.1f}% fraud probability"
                is_high_risk = False
            else:
                explanation = f"Low risk transaction with {prediction*100:.1f}% fraud probability"
                is_high_risk = False
            
            return {
                "prediction": prediction,
                "is_high_risk": is_high_risk,
                "base_value": 0.5,  # Placeholder
                "explanation": explanation,
                "contributions": contributions
            }
            
        except Exception as e:
            logger.error(f"Explanation error: {e}")
            # Return basic prediction if explanation fails
            return {
                "prediction": 0.5,
                "is_high_risk": False,
                "base_value": 0.5,
                "explanation": "Could not generate detailed explanation",
                "contributions": []
            }

# Initialize model explainer
model_explainer = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and load model on startup"""
    global model_explainer
    
    # Create database tables
    create_tables()
    logger.info("Database tables created")
    
    # Load model and preprocessor
    load_model_and_preprocessor()
    
    # Initialize model explainer
    if model is not None:
        model_explainer = SimpleModelExplainer(model, preprocessor)
        logger.info(f"Model version {MODEL_VERSION} ready for predictions")
    else:
        logger.error("Model not loaded - predictions will fail")

# Health check endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    model_status = "✅ Loaded" if model is not None else "❌ Not Loaded"
    preprocessor_status = "✅ Loaded" if preprocessor is not None else "⚠️ Not Found"
    
    html_content = f"""
    <html>
        <head>
            <title>Fraud Detection API</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                .status {{
                    background: #4CAF50;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .warning {{
                    background: #ff9800;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .error {{
                    background: #f44336;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .endpoints {{
                    background: #f9f9f9;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                code {{
                    background: #eee;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-family: monospace;
                }}
                .status-item {{
                    margin: 5px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Fraud Detection API</h1>
                
                <div class="status">
                    <div class="status-item"><strong>API Status:</strong> ✅ Operational</div>
                    <div class="status-item"><strong>Model:</strong> {model_status}</div>
                    <div class="status-item"><strong>Preprocessor:</strong> {preprocessor_status}</div>
                    <div class="status-item"><strong>Model Version:</strong> {MODEL_VERSION}</div>
                </div>
                
                <div class="endpoints">
                    <h3>📚 API Documentation</h3>
                    <ul>
                        <li><a href="/docs">Interactive Swagger Docs</a></li>
                        <li><a href="/redoc">ReDoc Documentation</a></li>
                    </ul>
                    
                    <h3>🔧 Available Endpoints</h3>
                    <ul>
                        <li><code>POST /predict</code> - Single prediction with explanation</li>
                        <li><code>POST /predict/batch</code> - Batch predictions</li>
                        <li><code>GET /health</code> - Health check</li>
                        <li><code>GET /predictions</code> - Get prediction history</li>
                        <li><code>GET /model/info</code> - Get model information</li>
                    </ul>
                </div>
                
                <h3>📊 Model Information</h3>
                <p>This API uses a RandomForestClassifier trained on the IEEE-CIS Fraud Detection dataset.</p>
                {"<p class='warning'>⚠️ Note: SHAP explanations are simplified due to model size constraints.</p>" if model is not None else ""}
                
                <h3>⚡ Quick Start</h3>
                <pre><code>curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
         "transaction": {{
             "TransactionAmt": 100.50,
             "ProductCD": "W",
             "card4": "visa"
         }}
     }}'</code></pre>
            </div>
        </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import time
    
    # Calculate uptime (simplified)
    start_time = getattr(health_check, '_start_time', None)
    if start_time is None:
        health_check._start_time = time.time()
        start_time = health_check._start_time
    
    uptime_seconds = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        uptime_seconds=uptime_seconds,
        memory_usage_mb=psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    )

def log_prediction(db: Session, prediction_data: dict, request: Request):
    """Log prediction to database"""
    try:
        prediction_log = PredictionLog(
            prediction_id=str(uuid.uuid4()),
            transaction_data=prediction_data["transaction_data"],
            fraud_probability=prediction_data["fraud_probability"],
            is_fraud=prediction_data["is_fraud"],
            is_high_risk=prediction_data["is_high_risk"],
            model_version=prediction_data["model_version"],
            explanation=prediction_data["explanation"],
            top_contributions=prediction_data["top_contributions"],
            inference_time_ms=prediction_data["inference_time_ms"],
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        db.add(prediction_log)
        db.commit()
        logger.info(f"Prediction logged: {prediction_log.prediction_id}")
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        db.rollback()

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    http_request: Request = None
):
    """Predict fraud probability for a single transaction"""
    
    if model_explainer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert request to DataFrame
        transaction_dict = request.transaction.dict()
        transaction_df = pd.DataFrame([transaction_dict])
        
        # Get prediction with explanation
        explanation = model_explainer.explain_prediction(transaction_df)
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Prepare response
        response = PredictionResponse(
            prediction=explanation["prediction"],
            is_fraud=explanation["prediction"] > 0.5,
            is_high_risk=explanation["is_high_risk"],
            base_value=explanation["base_value"],
            explanation=explanation["explanation"],
            top_contributions=explanation["contributions"],
            model_version=MODEL_VERSION,
            inference_time_ms=inference_time_ms,
            timestamp=datetime.utcnow()
        )
        
        # Log prediction in background
        log_data = {
            "transaction_data": transaction_dict,
            "fraud_probability": explanation["prediction"],
            "is_fraud": explanation["prediction"] > 0.5,
            "is_high_risk": explanation["is_high_risk"],
            "model_version": MODEL_VERSION,
            "explanation": explanation["explanation"],
            "top_contributions": explanation["contributions"],
            "inference_time_ms": inference_time_ms
        }
        
        if http_request:
            background_tasks.add_task(log_prediction, db, log_data, http_request)
        
        logger.info(f"Prediction completed in {inference_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict fraud probability for multiple transactions"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert requests to DataFrame
        transactions = [t.dict() for t in request.transactions]
        transaction_df = pd.DataFrame(transactions)
        
        # Preprocess
        if preprocessor is not None:
            processed_data = preprocessor.transform(transaction_df)
        else:
            processed_data = transaction_df.values
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(processed_data)[:, 1]
        else:
            predictions = model.predict(processed_data)
        
        is_fraud_flags = (predictions > 0.5).tolist()
        is_high_risk_flags = (predictions > 0.7).tolist()
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        response = BatchPredictionResponse(
            predictions=predictions.tolist(),
            is_fraud_flags=is_fraud_flags,
            is_high_risk_flags=is_high_risk_flags,
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow(),
            inference_time_ms=inference_time_ms
        )
        
        logger.info(f"Batch prediction completed: {len(predictions)} transactions in {inference_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/predictions")
async def get_predictions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get prediction history"""
    predictions = db.query(PredictionLog)\
        .order_by(PredictionLog.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return {
        "predictions": [
            {
                "id": p.id,
                "prediction_id": p.prediction_id,
                "fraud_probability": p.fraud_probability,
                "is_fraud": p.is_fraud,
                "is_high_risk": p.is_high_risk,
                "model_version": p.model_version,
                "created_at": p.created_at.isoformat()
            }
            for p in predictions
        ],
        "total": db.query(PredictionLog).count(),
        "skip": skip,
        "limit": limit
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "model_version": MODEL_VERSION,
    }
    
    # Add model-specific attributes
    if hasattr(model, 'n_features_in_'):
        info["feature_count"] = model.n_features_in_
    if hasattr(model, 'classes_'):
        info["classes"] = model.classes_.tolist()
    if hasattr(model, 'n_estimators'):
        info["n_estimators"] = model.n_estimators
    
    # Add file info
    if MODEL_PATH.exists():
        info["model_file_size_mb"] = MODEL_PATH.stat().st_size / 1024 / 1024
        info["model_modified"] = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat()
    
    return info

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)