from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np

class TransactionBase(BaseModel):
    """Base schema for transaction data"""
    TransactionAmt: float = Field(..., description="Transaction amount", example=100.50)
    ProductCD: str = Field(..., description="Product code", example="W")
    card1: Optional[float] = Field(None, description="Card identifier 1", example=13926.0)
    card2: Optional[float] = Field(None, description="Card identifier 2", example=226.0)
    card3: Optional[float] = Field(None, description="Card identifier 3", example=150.0)
    card4: Optional[str] = Field(None, description="Card type", example="visa")
    card5: Optional[float] = Field(None, description="Card identifier 5", example=226.0)
    card6: Optional[str] = Field(None, description="Card category", example="debit")
    addr1: Optional[float] = Field(None, description="Address 1", example=315.0)
    addr2: Optional[float] = Field(None, description="Address 2", example=87.0)
    dist1: Optional[float] = Field(None, description="Distance 1", example=75.0)
    dist2: Optional[float] = Field(None, description="Distance 2", example=80.0)
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain", example="gmail.com")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain", example="yahoo.com")
    C1: Optional[float] = Field(None, description="Count 1", example=1.0)
    C2: Optional[float] = Field(None, description="Count 2", example=1.0)
    C3: Optional[float] = Field(None, description="Count 3", example=1.0)
    C4: Optional[float] = Field(None, description="Count 4", example=0.0)
    C5: Optional[float] = Field(None, description="Count 5", example=0.0)
    C6: Optional[float] = Field(None, description="Count 6", example=0.0)
    C7: Optional[float] = Field(None, description="Count 7", example=0.0)
    C8: Optional[float] = Field(None, description="Count 8", example=0.0)
    C9: Optional[float] = Field(None, description="Count 9", example=0.0)
    C10: Optional[float] = Field(None, description="Count 10", example=0.0)
    C11: Optional[float] = Field(None, description="Count 11", example=1.0)
    C12: Optional[float] = Field(None, description="Count 12", example=0.0)
    C13: Optional[float] = Field(None, description="Count 13", example=0.0)
    C14: Optional[float] = Field(None, description="Count 14", example=0.0)
    D1: Optional[float] = Field(None, description="Delta days 1", example=14.0)
    D2: Optional[float] = Field(None, description="Delta days 2", example=14.0)
    D3: Optional[float] = Field(None, description="Delta days 3", example=14.0)
    D4: Optional[float] = Field(None, description="Delta days 4", example=14.0)
    D5: Optional[float] = Field(None, description="Delta days 5", example=14.0)
    D6: Optional[float] = Field(None, description="Delta days 6", example=14.0)
    D7: Optional[float] = Field(None, description="Delta days 7", example=14.0)
    D8: Optional[float] = Field(None, description="Delta days 8", example=14.0)
    D9: Optional[float] = Field(None, description="Delta days 9", example=14.0)
    D10: Optional[float] = Field(None, description="Delta days 10", example=14.0)
    D11: Optional[float] = Field(None, description="Delta days 11", example=14.0)
    D12: Optional[float] = Field(None, description="Delta days 12", example=14.0)
    D13: Optional[float] = Field(None, description="Delta days 13", example=14.0)
    D14: Optional[float] = Field(None, description="Delta days 14", example=14.0)
    D15: Optional[float] = Field(None, description="Delta days 15", example=14.0)
    M1: Optional[str] = Field(None, description="Match 1", example="T")
    M2: Optional[str] = Field(None, description="Match 2", example="T")
    M3: Optional[str] = Field(None, description="Match 3", example="T")
    M4: Optional[str] = Field(None, description="Match 4", example="M0")
    M5: Optional[str] = Field(None, description="Match 5", example="M0")
    M6: Optional[str] = Field(None, description="Match 6", example="M0")
    M7: Optional[str] = Field(None, description="Match 7", example="M0")
    M8: Optional[str] = Field(None, description="Match 8", example="M0")
    M9: Optional[str] = Field(None, description="Match 9", example="M0")
    DeviceType: Optional[str] = Field(None, description="Device type", example="desktop")
    TransactionDT: Optional[float] = Field(None, description="Transaction datetime", example=86400.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 100.50,
                "ProductCD": "W",
                "card1": 13926.0,
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "C1": 1.0,
                "C2": 1.0,
                "C3": 1.0,
                "D1": 14.0,
                "M1": "T",
                "DeviceType": "desktop",
                "TransactionDT": 86400.0
            }
        }

class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    transaction: TransactionBase
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction": {
                    "TransactionAmt": 100.50,
                    "ProductCD": "W",
                    "card1": 13926.0,
                    "card4": "visa",
                    "card6": "debit",
                    "P_emaildomain": "gmail.com"
                }
            }
        }

class Contribution(BaseModel):
    """Schema for SHAP contribution"""
    feature: str
    value: float
    contribution: float
    absolute_contribution: float

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: float = Field(..., description="Fraud probability (0-1)", example=0.85)
    is_fraud: bool = Field(..., description="Binary fraud prediction", example=True)
    is_high_risk: bool = Field(..., description="High risk flag", example=True)
    base_value: float = Field(..., description="SHAP base value", example=0.034)
    explanation: str = Field(..., description="Human-readable explanation")
    top_contributions: List[Contribution] = Field(..., description="Top feature contributions")
    model_version: str = Field(..., description="Model version", example="1.0.0")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds", example=45.2)
    timestamp: datetime = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="Service status", example="healthy")
    model_loaded: bool = Field(..., description="Whether model is loaded", example=True)
    model_version: str = Field(..., description="Model version", example="1.0.0")
    uptime_seconds: float = Field(..., description="Service uptime in seconds", example=3600.5)
    memory_usage_mb: float = Field(..., description="Memory usage in MB", example=512.3)

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    transactions: List[TransactionBase]
    
    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {"TransactionAmt": 100.50, "ProductCD": "W"},
                    {"TransactionAmt": 250.00, "ProductCD": "C"}
                ]
            }
        }

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[float]
    is_fraud_flags: List[bool]
    is_high_risk_flags: List[bool]
    model_version: str
    timestamp: datetime
    inference_time_ms: float