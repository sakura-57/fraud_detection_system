import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logger with consistent formatting"""
    
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_path / "fraud_detection.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger