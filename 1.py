# =========================
# PERFORMANCE PRACTICAL
# =========================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# Step 2: Load Sample Dataset
# =========================

iris = load_iris(as_frame=True)

df = iris.frame

print("Dataset Head:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# =========================
# Step 3: Perform EDA
# =========================

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nColumn Names:")
print(df.columns)

print("\nChecking Duplicate Rows:")
print(df.duplicated().sum())

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df, hue='target')
plt.show()

# =========================
# Step 4: Display Missing Values
# =========================

# Creating sample missing values
df.iloc[2, 0] = np.nan
df.iloc[10, 1] = np.nan

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

# Handling missing values using mean
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# =========================
# Step 5: Display and Handle Outliers
# =========================

# Boxplot before handling outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("Boxplot Before Outlier Handling")
plt.show()

# Using IQR Method
Q1 = df.quantile(0.25, numeric_only=True)
Q3 = df.quantile(0.75, numeric_only=True)

IQR = Q3 - Q1

# Removing outliers
df_outlier_removed = df[~((df < (Q1 - 1.5 * IQR)) | 
                          (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nShape Before Removing Outliers:", df.shape)
print("Shape After Removing Outliers:", df_outlier_removed.shape)

# Boxplot after handling outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=df_outlier_removed)
plt.title("Boxplot After Outlier Handling")
plt.show()

# =========================
# Step 6: Data Encoding
# =========================

# Converting target into categorical labels
df_outlier_removed['species'] = iris.target_names[
    df_outlier_removed['target'].astype(int)
]

print("\nBefore Encoding:")
print(df_outlier_removed[['species']].head())

# Label Encoding
encoder = LabelEncoder()

df_outlier_removed['species_encoded'] = encoder.fit_transform(
    df_outlier_removed['species']
)

print("\nAfter Encoding:")
print(df_outlier_removed[['species', 'species_encoded']].head())

# =========================
# Step 7: Feature Scaling
# =========================

scaler = StandardScaler()

features = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

print("\nBefore Scaling:")
print(df_outlier_removed[features].head())

# Applying scaling
scaled_data = scaler.fit_transform(df_outlier_removed[features])

scaled_df = pd.DataFrame(scaled_data, columns=features)

print("\nAfter Scaling:")
print(scaled_df.head())

# =========================
# End
# =========================
