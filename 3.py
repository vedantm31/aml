# ======================================================
# SEQUENTIAL FORWARD SELECTION AND BACKWARD ELIMINATION
# USING LOGISTIC REGRESSION
# ======================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# ======================================================
# Step 2: Load Dataset
# ======================================================

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

df['target'] = data.target

print("Dataset Head:")
print(df.head())

# ======================================================
# Step 3: Preprocessing
# ======================================================

X = df.drop('target', axis=1)
y = df['target']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================================
# Step 4: Apply Logistic Regression
# ======================================================

model = LogisticRegression(max_iter=5000)

# ======================================================
# Step 5: Sequential Forward Selection (SFS)
# ======================================================

print("\n========================================")
print("Sequential Forward Selection (SFS)")
print("========================================")

sfs_forward = SFS(
    model,
    k_features=5,
    forward=True,
    floating=False,
    scoring='accuracy',
    cv=5
)

sfs_forward.fit(X_train_scaled, y_train)

forward_features = list(sfs_forward.k_feature_names_)

print("\nTop Selected Features using SFS:")
print(forward_features)

# Train model using selected features
forward_indices = list(sfs_forward.k_feature_idx_)

model.fit(X_train_scaled[:, forward_indices], y_train)

y_pred_forward = model.predict(X_test_scaled[:, forward_indices])

forward_accuracy = accuracy_score(y_test, y_pred_forward)

print("\nAccuracy using SFS:", forward_accuracy)

# ======================================================
# Step 6: Sequential Backward Elimination (SBE)
# ======================================================

print("\n========================================")
print("Sequential Backward Elimination (SBE)")
print("========================================")

sfs_backward = SFS(
    model,
    k_features=5,
    forward=False,
    floating=False,
    scoring='accuracy',
    cv=5
)

sfs_backward.fit(X_train_scaled, y_train)

backward_features = list(sfs_backward.k_feature_names_)

print("\nTop Selected Features using SBE:")
print(backward_features)

# Train model using selected features
backward_indices = list(sfs_backward.k_feature_idx_)

model.fit(X_train_scaled[:, backward_indices], y_train)

y_pred_backward = model.predict(X_test_scaled[:, backward_indices])

backward_accuracy = accuracy_score(y_test, y_pred_backward)

print("\nAccuracy using SBE:", backward_accuracy)

# ======================================================
# Step 7: Compare Performance
# ======================================================

print("\n========================================")
print("Performance Comparison")
print("========================================")

comparison = pd.DataFrame({
    'Method': ['Sequential Forward Selection', 'Sequential Backward Elimination'],
    'Accuracy': [forward_accuracy, backward_accuracy]
})

print(comparison)

# ======================================================
# END
# ======================================================
