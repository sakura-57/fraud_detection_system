"""
Hugging Face Spaces deployment entry point
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.main import app

# This is needed for Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)