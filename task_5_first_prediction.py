import json
import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# 1. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 2. Ingestion & Stratified Split
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.1765, stratify=y_dev, random_state=SEED
)

# 3. Baseline Model (Majority Class)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_preds = dummy.predict(X_val)

dummy_metrics = {
    "model": "Dummy Baseline",
    "val_accuracy": float(accuracy_score(y_val, dummy_preds)),
    "val_recall_malignant": float(recall_score(y_val, dummy_preds, pos_label=0)),
    "val_f1_weighted": float(f1_score(y_val, dummy_preds, average="weighted"))
}

# 4. First Real Model Pipeline (StandardScaler + Logistic Regression)
first_model_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=SEED, max_iter=500))
])

first_model_pipe.fit(X_train, y_train)
val_preds = first_model_pipe.predict(X_val)
val_probs = first_model_pipe.predict_proba(X_val)[:, 1]

model_metrics = {
    "model": "Logistic Regression (StandardScaler)",
    "val_accuracy": float(accuracy_score(y_val, val_preds)),
    "val_recall_malignant": float(recall_score(y_val, val_preds, pos_label=0)),
    "val_precision_malignant": float(precision_score(y_val, val_preds, pos_label=0)),
    "val_f1_weighted": float(f1_score(y_val, val_preds, average="weighted")),
    "val_roc_auc": float(roc_auc_score(y_val, val_probs))
}

# 5. Output Comparisons
print("=== Baseline vs. First Model Performance ===")
comparison_df = pd.DataFrame([dummy_metrics, model_metrics]).fillna("-")
print(comparison_df.to_string(index=False))

# 6. Error Inspection (False Negatives on Malignant Cases)
errors_idx = (y_val == 0) & (val_preds == 1)
false_negatives = X_val[errors_idx]
print(f"\nTotal Malignant Cases Missed (False Negatives): {len(false_negatives)}")
if not false_negatives.empty:
    key_features = ["mean radius", "mean perimeter", "mean concave points"]
    print("Borderline feature values for misclassified cases:")
    print(false_negatives[key_features])

# 7. Persist to Experiment Log
log_entry = {
    "task": "Task 5 - The First Prediction",
    "baseline": dummy_metrics,
    "first_model": model_metrics,
    "false_negative_count": int(errors_idx.sum())
}

with open("task5_experiment_log.json", "w") as f:
    json.dump(log_entry, f, indent=2)
print("\nMetrics successfully appended to task5_experiment_log.json")