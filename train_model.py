# =========================================================
# Telco Churn Model Training Script (LOCAL / VS CODE)
# =========================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 1. LOAD DATASET
# =========================================================

DATA_PATH = "data/Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# =========================================================
# 2. CLEANING & PREPROCESSING
# =========================================================

# Drop customerID
df = df.drop(columns=["customerID"])

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing values
df = df.dropna()

# Encode target (WAJIB)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Separate features & target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# =========================================================
# 3. PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols)
    ]
)


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 5. MODELS & HYPERPARAMETERS
# =========================================================

models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, solver="liblinear"),
        "params": {
            "model__C": [0.1, 1, 10]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        }
    }
}

results = {}
summary = []

# =========================================================
# 6. TRAINING & EVALUATION
# =========================================================

for name, cfg in models.items():
    print(f"\nTraining {name}...")

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", cfg["model"])
        ]
    )

    grid = GridSearchCV(
        pipe,
        cfg["params"],
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "model": best_model,
        "best_params": grid.best_params_,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

    summary.append([name, acc, prec, rec, f1])

    print(f"{name} - F1 Score: {f1:.4f}")

# =========================================================
# 7. SELECT BEST MODEL
# =========================================================

summary_df = pd.DataFrame(
    summary,
    columns=["model", "accuracy", "precision", "recall", "f1_score"]
)

best_model_name = summary_df.sort_values(
    by="f1_score", ascending=False
).iloc[0]["model"]

best_model = results[best_model_name]["model"]

print("\nBEST MODEL:", best_model_name)
print(summary_df)

# =========================================================
# 8. SAVE ARTIFACTS (DEPLOYMENT READY)
# =========================================================

print("\nSaving model & artifacts...")

joblib.dump(best_model, "best_model.pkl")
joblib.dump(numerical_cols + categorical_cols, "feature_names.pkl")
joblib.dump(categorical_cols, "categorical_cols.pkl")

model_info = {
    "model_name": best_model_name,
    "best_params": results[best_model_name]["best_params"],
    "performance": {
        "accuracy": results[best_model_name]["accuracy"],
        "precision": results[best_model_name]["precision"],
        "recall": results[best_model_name]["recall"],
        "f1_score": results[best_model_name]["f1_score"]
    },
    "target_mapping": {"No": 0, "Yes": 1}
}

joblib.dump(model_info, "model_info.pkl")

print("✅ FILES CREATED:")
print(" - best_model.pkl")
print(" - feature_names.pkl")
print(" - categorical_cols.pkl")
print(" - model_info.pkl")

print("\nTRAINING COMPLETED SUCCESSFULLY 🎉")
