# ==========================================================
# DECISION TREE CLASSIFICATION MODEL
# ==========================================================

# ==========================================================
# STEP 1: IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ==========================================================
# STEP 2: LOAD DATASET
# ==========================================================

# Using Breast Cancer Dataset

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

print("Dataset Head:")
print(X.head())

# ==========================================================
# STEP 3: TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# STEP 4: BUILD DECISION TREE MODEL
# ==========================================================

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# ==========================================================
# STEP 5: MAKE PREDICTIONS
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# STEP 6: EVALUATE MODEL PERFORMANCE
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

# ==========================================================
# STEP 7: DISPLAY RESULTS
# ==========================================================

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print("\nAccuracy:")
print(accuracy)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1 Score:")
print(f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================================
# STEP 8: VISUALIZE DECISION TREE
# ==========================================================

plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True
)

plt.title("Decision Tree")

plt.show()

# ==========================================================
# STEP 9: INTERPRETATION OF RESULTS
# ==========================================================

print("\n===================================")
print("INTERPRETATION")
print("===================================")

print("""
1. Accuracy shows the overall correctness of the model.
2. Precision indicates how many predicted positive cases
   were actually positive.
3. Recall indicates how many actual positive cases
   were correctly identified.
4. F1 Score provides a balance between precision and recall.
5. Higher values of all metrics indicate better model performance.
6. The Decision Tree model successfully classified
   the dataset with good accuracy and prediction capability.
""")

# ==========================================================
# END
# ==========================================================
