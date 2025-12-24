import pandas as pd
import joblib
import shap
from pathlib import Path
import sys

PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP\Desktop\coding\fraud detection project"
)

MODEL_DIR = PROJECT_ROOT / "backend" / "model"
DATA_FILE = PROJECT_ROOT / "creditcard.csv"

print("\n📦 Loading model and scaler...")
model = joblib.load(MODEL_DIR / "model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


print("📊 Loading dataset...")
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Class", "Time"], errors="ignore")

X_scaled = scaler.transform(X)

print("🧠 Creating SHAP explainer...")


explainer = shap.Explainer(model, X_scaled)

SAMPLE_SIZE = 1000
shap_values = explainer(X_scaled[:SAMPLE_SIZE])

print("📈 Generating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X.iloc[:SAMPLE_SIZE],
    show=True
)

print("\n✅ SHAP analysis completed successfully")
