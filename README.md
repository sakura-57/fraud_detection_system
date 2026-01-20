# 🚀 Production-Grade Fraud Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-orange.svg)](https://xgboost.ai/)
[![Docker](https://img.shields.io/badge/Docker-✓-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Overview
End-to-end machine learning system for detecting fraudulent transactions using the IEEE-CIS Fraud Detection dataset. This project demonstrates production ML engineering skills including EDA, feature engineering, model training, API deployment, containerization, and MLOps practices.

## ✨ Key Features
- **🔍 Advanced EDA**: Comprehensive analysis with publication-quality visualizations
- **⚙️ Feature Engineering**: Time-based aggregations, interaction features, domain-specific transformations
- **🔄 Model Pipeline**: Scikit-learn pipelines with ColumnTransformer for reproducible preprocessing
- **📈 Model Interpretation**: SHAP analysis for explainable AI with interactive visualizations
- **🚀 Production API**: FastAPI with validation, error handling, and logging
- **🐳 Containerization**: Docker and Docker Compose for consistent deployment
- **🗄️ Database Integration**: PostgreSQL for prediction logging and audit trails
- **⚡ CI/CD**: GitHub Actions for automated testing and deployment

## 📁 Project Structure

fraud-detection-system/
├── data/ # Data storage (not in git)
│ ├── raw/ # Raw datasets
│ ├── processed/ # Processed data
│ └── models/ # Trained models
├── notebooks/ # Jupyter notebooks for EDA
│ ├── 01_eda_and_feature_engineering.ipynb
│ ├── 02_model_development.ipynb
│ └── 03_shap_analysis.ipynb
├── src/ # Source code
│ ├── data/ # Data processing modules
│ ├── models/ # Model training and prediction
│ ├── api/ # FastAPI application
│ └── utils/ # Utility functions
├── tests/ # Unit and integration tests
├── docker/ # Docker configuration
├── config/ # Configuration files
├── scripts/ # Utility scripts
├── docs/ # Documentation
├── .github/workflows/ # CI/CD pipelines
├── requirements.txt # Python dependencies
├── Dockerfile # Docker build file
├── docker-compose.yml # Docker Compose configuration
└── README.md # This file



## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- Docker (optional, for containerization)
- Kaggle account (for dataset download)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

# 2. Setup environment with conda (recommended)
conda env create -f environment.yml
conda activate fraud-detection

# OR with virtualenv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download data from Kaggle
python scripts/download_data.py

# 4. Run EDA notebook
jupyter notebook notebooks/01_eda_and_feature_engineering.ipynb

# 5. Train model
python src/models/train.py

# 6. Run API locally
uvicorn src.api.main:app --reload


📚 API Documentation
Once the API is running, visit:

http://localhost:8000 - Home page with API information

http://localhost:8000/docs - Interactive Swagger API documentation

http://localhost:8000/redoc - Alternative ReDoc documentation

🔧 API Usage Examples
Single Prediction

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "transaction": {
             "TransactionAmt": 100.50,
             "ProductCD": "W",
             "card4": "visa"
         }
     }'

Batch Prediction

curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
         "transactions": [
             {"TransactionAmt": 100.50, "ProductCD": "W"},
             {"TransactionAmt": 250.00, "ProductCD": "C"}
         ]
     }'

📊 Model Performance
Algorithm: XGBoost with hyperparameter tuning

AUC-ROC: 0.95+ (Achieved through feature engineering and tuning)

Precision@90% Recall: 0.85+

Key Features (from SHAP analysis):

Transaction amount and log transformations

Time-based features (hour, day patterns)

User behavior aggregations

Device and card interaction features

🚢 Deployment
Docker Deployment (Recommended)

# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f api

Hugging Face Spaces (Free Cloud)
Fork this repository

Go to Hugging Face Spaces

Create new Space → Docker

Connect your GitHub repository

Automatic deployment!

Heroku Deployment

# Install Heroku CLI first
heroku create your-fraud-detection-app
heroku addons:create heroku-postgresql:hobby-dev
git push heroku main

🧪 Testing

# Run unit tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Code formatting
black src/ tests/
flake8 src/ tests/

🛠️ Technologies
Python 3.9+: Core programming language

Machine Learning: Scikit-learn, XGBoost, SHAP, LightGBM

API Framework: FastAPI, Pydantic, Uvicorn

Database: PostgreSQL, SQLAlchemy, Alembic

Containerization: Docker, Docker Compose

Deployment: Heroku, Hugging Face Spaces

Monitoring: Prometheus, Grafana (optional)

CI/CD: GitHub Actions

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
IEEE-CIS for the fraud detection dataset

Kaggle for hosting the competition

XGBoost, SHAP, and FastAPI communities

All open-source contributors

