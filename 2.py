# ==========================================
# FEATURE SELECTION USING 5 FILTER METHODS
# ==========================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif
)

# ==========================================
# Step 2: Load Dataset
# ==========================================

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

df['target'] = data.target

print("Dataset Head:")
print(df.head())

# ==========================================
# Step 3: Preprocess Dataset
# ==========================================

# Separate Features and Target
X = df.drop('target', axis=1)
y = df['target']

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Dataset Head:")
print(X_scaled.head())

# ==========================================
# FILTER METHOD 1:
# Variance Threshold
# ==========================================

print("\n===================================")
print("1. Variance Threshold")
print("===================================")

vt = VarianceThreshold(threshold=0.5)

vt.fit(X_scaled)

variance_features = X.columns[vt.get_support()]

print("Selected Features:")
print(list(variance_features))

# ==========================================
# FILTER METHOD 2:
# Correlation Method
# ==========================================

print("\n===================================")
print("2. Correlation Method")
print("===================================")

corr_matrix = df.corr()

target_corr = corr_matrix['target'].abs().sort_values(ascending=False)

top_corr_features = target_corr[1:11]

print("Top Correlated Features:")
print(top_corr_features)

# ==========================================
# FILTER METHOD 3:
# Chi-Square Test
# ==========================================

print("\n===================================")
print("3. Chi-Square Test")
print("===================================")

# Chi2 requires non-negative values
X_positive = X - X.min()

chi_selector = SelectKBest(score_func=chi2, k=10)

chi_selector.fit(X_positive, y)

chi_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': chi_selector.scores_
})

chi_scores = chi_scores.sort_values(by='Score', ascending=False)

print("Top Features:")
print(chi_scores.head(10))

# ==========================================
# FILTER METHOD 4:
# ANOVA F-Test
# ==========================================

print("\n===================================")
print("4. ANOVA F-Test")
print("===================================")

anova_selector = SelectKBest(score_func=f_classif, k=10)

anova_selector.fit(X_scaled, y)

anova_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': anova_selector.scores_
})

anova_scores = anova_scores.sort_values(by='Score', ascending=False)

print("Top Features:")
print(anova_scores.head(10))

# ==========================================
# FILTER METHOD 5:
# Mutual Information
# ==========================================

print("\n===================================")
print("5. Mutual Information")
print("===================================")

mi_scores = mutual_info_classif(X_scaled, y)

mi_df = pd.DataFrame({
    'Feature': X.columns,
    'Score': mi_scores
})

mi_df = mi_df.sort_values(by='Score', ascending=False)

print("Top Features:")
print(mi_df.head(10))

# ==========================================
# END
# ==========================================
