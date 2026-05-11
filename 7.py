# ==========================================================
# LOAN APPROVAL PREDICTION USING SVM
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

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# ==========================================================
# STEP 2: LOAD DATASET
# ==========================================================

# Download Dataset:
# https://www.kaggle.com/datasets/ninzaami/loan-predication

# Replace with your file path
df = pd.read_csv("loan_prediction.csv")

print("Dataset Head:")
print(df.head())

# ==========================================================
# STEP 3: DATA PREPROCESSING
# ==========================================================

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values

for column in df.columns:

    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)

    else:
        df[column].fillna(df[column].mean(), inplace=True)

# ==========================================================
# STEP 4: ENCODE CATEGORICAL VARIABLES
# ==========================================================

encoder = LabelEncoder()

for column in df.columns:

    if df[column].dtype == 'object':
        df[column] = encoder.fit_transform(df[column])

# ==========================================================
# STEP 5: VISUALIZE DATASET
# ==========================================================

# Count plot of Loan Status
plt.figure(figsize=(6,4))

sns.countplot(x='Loan_Status', data=df)

plt.title("Loan Approval Distribution")

plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

# ==========================================================
# STEP 6: SPLIT FEATURES AND TARGET
# ==========================================================

X = df.drop('Loan_Status', axis=1)

y = df['Loan_Status']

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
# STEP 9: TRAIN SVM MODELS
# ==========================================================

models = {

    "Linear Kernel": SVC(
        kernel='linear',
        C=1
    ),

    "RBF Kernel": SVC(
        kernel='rbf',
        C=1,
        gamma='scale'
    ),

    "Polynomial Kernel": SVC(
        kernel='poly',
        degree=3,
        C=1,
        gamma='scale'
    )
}

results = []

# ==========================================================
# STEP 10: TRAIN AND EVALUATE MODELS
# ==========================================================

for name, model in models.items():

    print("\n===================================")
    print(name)
    print("===================================")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    # Store Results
    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    # Print Results
    print("\nConfusion Matrix:")
    print(cm)

    print("\nAccuracy:")
    print(accuracy)

    print("\nPrecision:")
    print(precision)

    print("\nRecall:")
    print(recall)

    print("\nF1 Score:")
    print(f1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ==========================================================
# STEP 11: COMPARE KERNEL PERFORMANCE
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=[
        'Kernel',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score'
    ]
)

print("\n===================================")
print("KERNEL PERFORMANCE COMPARISON")
print("===================================")

print(results_df)

# ==========================================================
# STEP 12: BEST MODEL
# ==========================================================

best_model = results_df.sort_values(
    by='Accuracy',
    ascending=False
).iloc[0]

print("\n===================================")
print("BEST PERFORMING MODEL")
print("===================================")

print(best_model)

# ==========================================================
# END
# ==========================================================
