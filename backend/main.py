import os
import requests
import numpy as np
import pandas as pd
import joblib
from io import StringIO, BytesIO
from fastapi import FastAPI, HTTPException

app = FastAPI(title="PaySafe UPI Fraud Detection API")

# -----------------------------
# Utility functions
# -----------------------------
def load_csv_from_drive(file_id: str) -> pd.DataFrame:
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

def load_model_from_drive(file_id: str):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# -----------------------------
# Load datasets and models
# -----------------------------
cleaned_data = load_csv_from_drive(os.getenv("CLEANED_DATA_ID"))
metrics_summary = load_csv_from_drive(os.getenv("METRICS_SUMMARY_ID"))

scaler = load_model_from_drive(os.getenv("SCALER_ID"))
isolation_forest = load_model_from_drive(os.getenv("ISOLATION_FOREST_ID"))
xgb_model = load_model_from_drive(os.getenv("XGB_MODEL_ID"))

shap_transaction_values = load_csv_from_drive(os.getenv("SHAP_TRANSACTION_VALUES_ID"))
shap_feature_importance_bp = load_csv_from_drive(os.getenv("SHAP_FEATURE_IMPORTANCE_BP_ID"))
hybrid_eval_test = load_csv_from_drive(os.getenv("HYBRID_EVAL_TEST_ID"))
shap_fraud_vs_non_fraud = load_csv_from_drive(os.getenv("SHAP_FRAUD_VS_NON_FRAUD_ID"))

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "PaySafe UPI Fraud Detection API is running!"}

@app.get("/preview")
def preview(rows: int = 5):
    return cleaned_data.head(rows).to_dict(orient="records")

@app.get("/metrics")
def get_metrics():
    return metrics_summary.to_dict(orient="records")

@app.get("/shap/importance")
def shap_importance(rows: int = 10):
    return shap_feature_importance_bp.head(rows).to_dict(orient="records")

@app.get("/shap/transactions")
def shap_transactions(rows: int = 10):
    return shap_transaction_values.head(rows).to_dict(orient="records")

@app.get("/shap/fraud_vs_nonfraud")
def shap_fraud(rows: int = 10):
    return shap_fraud_vs_non_fraud.head(rows).to_dict(orient="records")

@app.get("/hybrid_eval")
def hybrid_eval(rows: int = 10):
    return hybrid_eval_test.head(rows).to_dict(orient="records")

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(features: dict):
    try:
        print("Received raw input:", features)

        # Full feature engineering
        amount = features["amount"]
        log_amount = np.log1p(amount)
        hour = features.get("Hour", 14)
        daily_txn = features.get("DailyTxnCount", 3)
        orig_txn = features.get("OrigTxnCount", 2)
        dest_txn = features.get("DestTxnCount", 2)
        txn_window = features.get("TxnCountWindow", 5)

        # Optional engineered features
        is_night = 1 if hour < 6 or hour > 22 else 0
        rule_high = features.get("RuleHighValue", 0)
        rule_rapid = features.get("RuleRapidFire", 0)
        type_encoded = 1 if features.get("transaction_type") == "upi" else 0

        # Build full feature dictionary
        full_features = {
            "amount": amount,
            "LogAmount": log_amount,
            "Hour": hour,
            "DailyTxnCount": daily_txn,
            "OrigTxnCount": orig_txn,
            "DestTxnCount": dest_txn,
            "TxnCountWindow": txn_window,
            "IsNight": is_night,
            "RuleHighValue": rule_high,
            "RuleRapidFire": rule_rapid,
            "Type_encoded": type_encoded
        }

        # Filter to match scaler's expected columns
        expected_cols = scaler.feature_names_in_
        filtered_features = {col: full_features[col] for col in expected_cols}

        X = pd.DataFrame([filtered_features])
        print("Final input to model:", X)

        # Predict
        X_scaled = scaler.transform(X)
        anomaly_score = isolation_forest.decision_function(X_scaled)
        prediction = xgb_model.predict(X_scaled)

        return {
            "prediction": int(prediction[0]),
            "anomaly_score": float(anomaly_score[0])
        }

    except Exception as e:
        import traceback
        print("🔥 Internal error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
