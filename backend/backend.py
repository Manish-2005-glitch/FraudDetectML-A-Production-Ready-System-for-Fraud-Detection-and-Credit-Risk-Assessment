from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP\Desktop\coding\fraud detection project"
)

MODEL_DIR = PROJECT_ROOT / "backend" / "model"

model_path = MODEL_DIR / "model.pkl"
scaler_path = MODEL_DIR / "scaler.pkl"
threshold_path = MODEL_DIR / "threshold.txt"


print("📦 Loading model, scaler, threshold...")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
threshold = float(threshold_path.read_text())

print(f"✅ Threshold loaded: {threshold:.6f}")


app = FastAPI(
    title="Fraud Detection API",
    description="ML-powered fraud detection inference service",
    version="1.0"
)

class Transaction(BaseModel):
    features: list[float]



@app.get("/")
def health():
    return {
        "status": "running",
        "model_loaded": True,
        "threshold": threshold
    }


@app.post("/predict")
def predict(transaction: Transaction):
    print("🔥 /predict HIT")

    features = transaction.features
    
    if len(features) != 29:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 29 features, got {len(features)}"
        )

    try:
        X = np.array(features, dtype=float).reshape(1, -1)

        if hasattr(scaler, "n_features_in_"):
            print(f"🧐 Scaler expects: {scaler.n_features_in_} features")
            print(f"📨 We provided: {X.shape[1]} features")

        # ... inside predict function ...
        
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]

        is_fraud = bool(prob >= threshold)
        prob_value = float(prob)
        
        return {
            "fraud_probability": prob_value,
            "fraud": is_fraud,
            "threshold_used": threshold
        }

    except Exception as e:
        import traceback
        traceback.print_exc() 
        print(f"❌ CRITICAL ERROR: {e}")
        
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

