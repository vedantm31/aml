# ==========================================================
# PUNE HOUSE PRICE PREDICTION USING MACHINE LEARNING
# ==========================================================

# ==========================================================
# STEP 1: IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# ==========================================================
# STEP 2: LOAD DATASET
# ==========================================================

# Download dataset from:
# https://www.kaggle.com/datasets/altavish/boston-housing-dataset

# Replace with your file path
df = pd.read_csv("HousingData.csv")

print("Dataset Head:")
print(df.head())

# ==========================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# STEP 4: HANDLE MISSING VALUES
# ==========================================================

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

# Fill missing values using column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# ==========================================================
# STEP 5: HANDLE OUTLIERS USING IQR METHOD
# ==========================================================

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

# Remove outliers
df = df[
    ~((df < (Q1 - 1.5 * IQR)) |
      (df > (Q3 + 1.5 * IQR))).any(axis=1)
]

print("\nDataset Shape After Outlier Removal:")
print(df.shape)

# ==========================================================
# STEP 6: FEATURES AND TARGET
# ==========================================================

X = df.drop("MEDV", axis=1)   # MEDV = House Price
y = df["MEDV"]

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
# STEP 9: TRAIN MULTIPLE REGRESSION MODELS
# ==========================================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, mse, r2])

    print("\n===================================")
    print(name)
    print("===================================")

    print("MAE :", mae)
    print("MSE :", mse)
    print("R2 Score :", r2)

# ==========================================================
# STEP 10: COMPARE MODEL PERFORMANCE
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "MSE", "R2 Score"]
)

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(results_df)

# ==========================================================
# STEP 11: IDENTIFY BEST MODEL
# ==========================================================

best_model = results_df.sort_values(
    by="R2 Score",
    ascending=False
).iloc[0]

print("\n===================================")
print("BEST PERFORMING MODEL")
print("===================================")

print(best_model)

# ==========================================================
# STEP 12: FEATURE IMPORTANCE
# ==========================================================

# Using Random Forest for Feature Importance

rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train, y_train)

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===================================")
print("FEATURE IMPORTANCE")
print("===================================")

print(feature_importance)

# Plot Feature Importance
plt.figure(figsize=(10,6))

sns.barplot(
    x=feature_importance["Importance"],
    y=feature_importance["Feature"]
)

plt.title("Feature Importance")
plt.show()

# ==========================================================
# END
# ==========================================================
