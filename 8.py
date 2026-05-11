# ==========================================================
# BANK MARKETING TERM DEPOSIT PREDICTION
# ==========================================================

# ==========================================================
# STEP 1: IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ==========================================================
# STEP 2: LOAD DATASET
# ==========================================================

# Download Dataset From:
# https://archive.ics.uci.edu/dataset/222/bank+marketing

# Use bank-additional-full.csv file

# Replace with your dataset path
df = pd.read_csv(
    "bank-additional-full.csv",
    sep=';'
)

print("Dataset Head:")
print(df.head())

# ==========================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================================
# STEP 4: VISUALIZATION
# ==========================================================

# Target Variable Distribution
plt.figure(figsize=(6,4))

sns.countplot(x='y', data=df)

plt.title("Term Deposit Subscription")

plt.show()

# Age Distribution
plt.figure(figsize=(8,5))

sns.histplot(df['age'], bins=30, kde=True)

plt.title("Age Distribution")

plt.show()

# Job vs Subscription
plt.figure(figsize=(12,5))

sns.countplot(x='job', hue='y', data=df)

plt.xticks(rotation=45)

plt.title("Job vs Subscription")

plt.show()

# ==========================================================
# STEP 5: MANUAL FEATURE SELECTION
# ==========================================================

# Selected Important Features after EDA

selected_features = [
    'age',
    'job',
    'marital',
    'education',
    'housing',
    'loan',
    'contact',
    'campaign',
    'previous',
    'poutcome',
    'emp.var.rate',
    'cons.price.idx',
    'euribor3m'
]

X = df[selected_features]

y = df['y']

# ==========================================================
# STEP 6: ENCODE CATEGORICAL VARIABLES
# ==========================================================

encoder = LabelEncoder()

for column in X.columns:

    if X[column].dtype == 'object':
        X[column] = encoder.fit_transform(X[column])

# Encode target variable
y = encoder.fit_transform(y)

# ==========================================================
# STEP 7: FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# STEP 8: TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# STEP 9: APPLY MACHINE LEARNING MODEL
# ==========================================================

# Using Random Forest Classifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ==========================================================
# STEP 10: MODEL EVALUATION
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================================
# STEP 11: FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n===================================")
print("FEATURE IMPORTANCE")
print("===================================")

print(importance)

# Plot Feature Importance
plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")

plt.show()

# ==========================================================
# STEP 12: CONCLUSION
# ==========================================================

print("\n===================================")
print("CONCLUSION")
print("===================================")

print("""
1. Random Forest Classifier was used for prediction.
2. Selected features were chosen manually after EDA.
3. The model predicts whether a customer subscribes
   to the term deposit plan or not.
4. Important influencing features include:
   euribor3m, campaign, age, and emp.var.rate.
""")

# ==========================================================
# END
# ==========================================================
