from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL - supports both PostgreSQL and SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    """Database model for logging predictions"""
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String, unique=True, index=True)
    transaction_data = Column(JSON, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    is_high_risk = Column(Boolean, nullable=False)
    model_version = Column(String, nullable=False)
    explanation = Column(Text, nullable=True)
    top_contributions = Column(JSON, nullable=True)
    inference_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<PredictionLog(id={self.id}, probability={self.fraud_probability:.3f})>"

class ModelVersion(Base):
    """Database model for tracking model versions"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    model_type = Column(String, nullable=False)
    performance_metrics = Column(JSON, nullable=True)
    feature_count = Column(Integer, nullable=False)
    trained_at = Column(DateTime, nullable=False)
    deployed_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<ModelVersion(version={self.version}, active={self.is_active})>"

# Create tables
def create_tables():
    """Create database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()