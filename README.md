# Fraud Detection System

An end-to-end machine learning system to detect fraudulent financial transactions. This project demonstrates the full pipeline from exploratory data analysis and model training to API deployment and model interpretation.

## 🚀 Features

*   **Exploratory Data Analysis:** Comprehensive analysis of the IEEE-CIS Fraud Detection dataset.
*   **Model Training:** Benchmarking of multiple algorithms (Logistic Regression, Random Forest, XGBoost) with hyperparameter tuning.
*   **Model Interpretation:** Usage of SHAP (SHapley Additive exPlanations) to explain model predictions.
*   **Production API:** A RESTful API built with FastAPI to serve predictions.
*   **Containerization:** Dockerized application for easy deployment.
*   **Live Deployment:** Publicly accessible API deployed on Heroku/Hugging Face Spaces.

## 📁 Project Structure

    ├── ...
    └── ...

## 🛠️ Installation & Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/sakura-57/fraud_detection_system.git
    cd fraud-detection-system
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

*(We will fill in more details as the project progresses)*

## 📊 Results

*   **Best Model:** XGBoost achieved an AUC-ROC score of 0.95 on the validation set.
*   **Key Features:** Transaction amount, frequency, and specific card-related features were most predictive of fraud.

## 📝 License

This project is licensed under the MIT License.