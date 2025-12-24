import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP\Desktop\coding\fraud detection project"
)

MODEL_DIR = PROJECT_ROOT / "backend" / "model"
DATA_FILE = PROJECT_ROOT / "creditcard.csv"

print("\n📦 Loading scaler...")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

print("📊 Loading dataset...")
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Class", "Time"], errors="ignore")


# =============================
reference = X.sample(frac=0.5, random_state=42)
current = X.drop(reference.index)

reference_scaled = scaler.transform(reference)
current_scaled = scaler.transform(current)


def calculate_psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = np.sum(
        (expected_perc - actual_perc)
        * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
    )

    return psi

print("\n📉 Calculating feature drift (PSI)...")

psi_scores = {}

for i, feature in enumerate(X.columns):
    psi_value = calculate_psi(
        reference_scaled[:, i],
        current_scaled[:, i]
    )
    psi_scores[feature] = psi_value

psi_df = (
    pd.DataFrame.from_dict(psi_scores, orient="index", columns=["PSI"])
    .sort_values("PSI", ascending=False)
)

print("\n📊 Top drifting features:")
print(psi_df.head(10))

overall_psi = psi_df["PSI"].mean()

print("\n🔍 Overall Drift Score (Mean PSI):", round(overall_psi, 4))

# Drift interpretation
if overall_psi < 0.10:
    status = "✅ No significant drift"
elif overall_psi < 0.20:
    status = "⚠️ Moderate drift – monitor closely"
else:
    status = "🚨 High drift – retraining recommended"

print("📢 Drift Status:", status)
