import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve
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
y = df["Class"]


X_scaled = scaler.transform(X)


print("🔮 Predicting fraud probabilities...")
probs = model.predict_proba(X_scaled)[:, 1]

precision, recall, thresholds = precision_recall_curve(y, probs)

thresholds = thresholds[:len(precision)]
scores = precision * recall
best_idx = scores.argmax()
best_threshold = thresholds[best_idx]


MODEL_DIR.mkdir(parents=True, exist_ok=True)
threshold_path = MODEL_DIR / "threshold.txt"
threshold_path.write_text(str(best_threshold))

print("\n✅ Precision–Recall Threshold Computed Successfully")
print(f"🎯 Best Threshold: {best_threshold:.6f}")

