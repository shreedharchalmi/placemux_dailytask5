import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Reproducibility
SEED = 42
np.random.seed(SEED)

# 2. Ingestion & Synthetic Categorical Injection for Protocol Demonstration
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

# Add a sample categorical column to demonstrate complete ColumnTransformer behavior
df['biopsy_source'] = np.random.choice(['Core Needle', 'FNA', 'Surgical'], size=len(df))

X = df.drop(columns=['target'])
y = df['target']

# 3. Stratified Train / Validation / Test Split
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.1765, stratify=y_dev, random_state=SEED
)

# 4. Feature Group Identification
num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# 5. Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# 6. Fit ONLY on Training Data (Prevent Leakage)
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# 7. Verification Checks
print("=== Verification & Integrity Checks ===")
print(f"X_train Transformed Shape: {X_train_processed.shape}")
print(f"X_val Transformed Shape:   {X_val_processed.shape}")
print(f"Any NaN in Train Processed: {np.isnan(X_train_processed).any()}")
print(f"Any NaN in Val Processed:   {np.isnan(X_val_processed).any()}")

# 8. Save Fitted Preprocessor for Serving
artifact_path = "fitted_preprocessor.joblib"
joblib.dump(preprocessor, artifact_path)
print(f"\nSuccessfully serialized preprocessor to: {artifact_path}")