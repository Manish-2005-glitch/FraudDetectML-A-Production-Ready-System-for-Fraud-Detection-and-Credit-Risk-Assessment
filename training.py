import optuna
import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


df = pd.read_csv("creditcard.csv")
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

print("✅ Data loaded & split completed")


# Optuna Objective

def objective(trial):
    start_time = time.time()

    model_type = trial.suggest_categorical(
        "model", ["lr", "rf", "xgb"]
    )

    print(f"\n🚀 Trial {trial.number} STARTED")
    print(f"🔍 Model selected: {model_type.upper()}")

    # ---------- Logistic Regression ----------
    if model_type == "lr":
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-4, 1e2, log=True),
            solver=trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
            class_weight="balanced",
            max_iter=1000
        )

    # ---------- Random Forest ----------
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 5, 25),
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    # ---------- XGBoost ----------
    else:
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 400),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )

   
    print("⏳ Training started...")
    model.fit(X_train, y_train)

    
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    elapsed = time.time() - start_time

    print(f"✅ Trial {trial.number} FINISHED")
    print(f"📈 ROC-AUC: {auc:.5f}")
    print(f"⏱ Time taken: {elapsed:.2f} seconds")

    return auc

# Run Optuna

print("\n🚦 Starting Optuna Optimization...\n")

study = optuna.create_study(direction="maximize")
study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)

print("\n🏆 OPTUNA FINISHED")
print("Best ROC-AUC:", study.best_value)
print("Best Params:", study.best_params)


# Train Final Model

print("\n🚀 Training final best model on full data")

params = study.best_params.copy()
model_type = params.pop("model")

if model_type == "lr":
    final_model = LogisticRegression(
        **params,
        class_weight="balanced",
        max_iter=1000
    )
elif model_type == "rf":
    final_model = RandomForestClassifier(
        **params,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
else:
    final_model = XGBClassifier(
        **params,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

final_model.fit(X_train, y_train)


# Overfitting Check
train_auc = roc_auc_score(
    y_train, final_model.predict_proba(X_train)[:, 1]
)
test_auc = roc_auc_score(
    y_test, final_model.predict_proba(X_test)[:, 1]
)

print("\n🔍 Overfitting Check")
print("Train ROC-AUC:", train_auc)
print("Test  ROC-AUC:", test_auc)

# =============================
PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP\Desktop\coding\fraud detection project"
)

MODEL_DIR = PROJECT_ROOT / "backend" / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(final_model, MODEL_DIR / "model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

print("\n💾 Model & scaler saved successfully")
