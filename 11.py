# ==========================================================
# CUSTOMER PURCHASE PREDICTION USING ENSEMBLE METHODS
# ==========================================================

# ==========================================================
# STEP 1: IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ==========================================================
# STEP 2: CREATE SAMPLE CUSTOMER PURCHASE DATASET
# ==========================================================

X, y = make_classification(
    n_samples=1000,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    random_state=42
)

# Create DataFrame
df = pd.DataFrame(
    X,
    columns=[
        'Age',
        'Income',
        'Browsing_Time',
        'Pages_Visited',
        'Previous_Purchases',
        'Ad_Clicks'
    ]
)

df['Purchase'] = y

print("Dataset Head:")
print(df.head())

# ==========================================================
# STEP 3: DATA PREPROCESSING
# ==========================================================

X = df.drop('Purchase', axis=1)

y = df['Purchase']

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# STEP 4: TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# STEP 5: BAGGING USING DECISION TREE
# ==========================================================

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging_model.fit(X_train, y_train)

bagging_pred = bagging_model.predict(X_test)

# ==========================================================
# STEP 6: ADABOOST CLASSIFIER
# ==========================================================

adaboost_model = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

adaboost_model.fit(X_train, y_train)

adaboost_pred = adaboost_model.predict(X_test)

# ==========================================================
# STEP 7: STACKING MODEL
# ==========================================================

estimators = [
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression())
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stacking_model.fit(X_train, y_train)

stacking_pred = stacking_model.predict(X_test)

# ==========================================================
# STEP 8: EVALUATE MODELS
# ==========================================================

models = {
    "Bagging Decision Tree": bagging_pred,
    "AdaBoost Classifier": adaboost_pred,
    "Stacking Model": stacking_pred
}

results = []

for name, prediction in models.items():

    accuracy = accuracy_score(y_test, prediction)

    results.append([name, accuracy])

    print("\n===================================")
    print(name)
    print("===================================")

    print("\nAccuracy:")
    print(accuracy)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report:")
    print(classification_report(y_test, prediction))

# ==========================================================
# STEP 9: COMPARE MODEL PERFORMANCE
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=['Model', 'Accuracy']
)

print("\n===================================")
print("MODEL PERFORMANCE COMPARISON")
print("===================================")

print(results_df)

# ==========================================================
# STEP 10: VISUALIZE ACCURACY
# ==========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x='Model',
    y='Accuracy',
    data=results_df
)

plt.title("Accuracy Comparison of Ensemble Models")

plt.ylim(0,1)

plt.show()

# ==========================================================
# STEP 11: CONCLUSION
# ==========================================================

print("\n===================================")
print("CONCLUSION")
print("===================================")

print("""
1. Bagging improves stability and reduces overfitting.
2. AdaBoost improves weak learners sequentially.
3. Stacking combines multiple models to improve prediction.
4. The model with highest accuracy performs best for customer purchase prediction.
""")

# ==========================================================
# END
# ==========================================================
