# Create fix_preprocessor.py
import joblib
import yaml
from pathlib import Path
from src.data.preprocessor import FraudDataPreprocessor

print("Fixing preprocessor save format...")

# Load the current (broken) preprocessor
preprocessor_path = Path("data/models/preprocessor.joblib")
if preprocessor_path.exists():
    loaded = joblib.load(preprocessor_path)
    print(f"Current type: {type(loaded)}")
    
    if isinstance(loaded, dict):
        # Extract the actual preprocessor
        actual_preprocessor = loaded.get('preprocessor')
        if actual_preprocessor is not None:
            print("Found preprocessor in dict, re-saving...")
            
            # Save it directly
            joblib.dump(actual_preprocessor, "data/models/preprocessor_fixed.joblib")
            print("Saved fixed preprocessor to data/models/preprocessor_fixed.joblib")
            
            # Also save the dict wrapper if you want both
            joblib.dump(loaded, "data/models/preprocessor_full.joblib")
            print("Saved full dict to data/models/preprocessor_full.joblib")
        else:
            print("No preprocessor found in dict!")
    else:
        print("Preprocessor is already an object, not a dict")
else:
    print("Preprocessor file not found!")

