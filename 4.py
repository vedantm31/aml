# ==========================================================
# PCA AND LDA COMPARISON USING SCIKIT-LEARN
# ==========================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================================
# Step 2: Load Dataset
# ==========================================================

wine = load_wine()

X = wine.data
y = wine.target

feature_names = wine.feature_names

df = pd.DataFrame(X, columns=feature_names)

df['target'] = y

print("Dataset Head:")
print(df.head())

# ==========================================================
# Step 3: Preprocessing
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# Step 4: Apply PCA
# ==========================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\n===================================")
print("PCA Reduced Features")
print("===================================")

pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])

print(pca_df.head())

# ==========================================================
# Step 5: Apply LDA
# ==========================================================

lda = LDA(n_components=2)

X_lda = lda.fit_transform(X_scaled, y)

print("\n===================================")
print("LDA Reduced Features")
print("===================================")

lda_df = pd.DataFrame(X_lda, columns=['LDA1', 'LDA2'])

print(lda_df.head())

# ==========================================================
# Step 6: Train-Test Split
# ==========================================================

X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)

X_train_lda, X_test_lda, _, _ = train_test_split(
    X_lda,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# Step 7: Classification using Logistic Regression
# ==========================================================

model_pca = LogisticRegression(max_iter=5000)

model_pca.fit(X_train_pca, y_train)

y_pred_pca = model_pca.predict(X_test_pca)

pca_accuracy = accuracy_score(y_test, y_pred_pca)

# ----------------------------------------------------------

model_lda = LogisticRegression(max_iter=5000)

model_lda.fit(X_train_lda, y_train)

y_pred_lda = model_lda.predict(X_test_lda)

lda_accuracy = accuracy_score(y_test, y_pred_lda)

# ==========================================================
# Step 8: Compare Accuracy
# ==========================================================

print("\n===================================")
print("Classification Accuracy Comparison")
print("===================================")

comparison = pd.DataFrame({
    'Method': ['PCA', 'LDA'],
    'Accuracy': [pca_accuracy, lda_accuracy]
})

print(comparison)

# ==========================================================
# Step 9: Plot 2D Projection after PCA
# ==========================================================

plt.figure(figsize=(8,6))

for label in np.unique(y):
    plt.scatter(
        X_pca[y == label, 0],
        X_pca[y == label, 1],
        label=wine.target_names[label]
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("2D Projection using PCA")
plt.legend()

plt.show()

# ==========================================================
# Step 10: Plot 2D Projection after LDA
# ==========================================================

plt.figure(figsize=(8,6))

for label in np.unique(y):
    plt.scatter(
        X_lda[y == label, 0],
        X_lda[y == label, 1],
        label=wine.target_names[label]
    )

plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.title("2D Projection using LDA")
plt.legend()

plt.show()

# ==========================================================
# Step 11: Comments on Visual Differences
# ==========================================================

print("\n===================================")
print("Comments on Visual Differences")
print("===================================")

print("""
1. PCA focuses on maximizing variance in the dataset.
2. PCA does not consider class labels.
3. LDA focuses on maximizing class separability.
4. LDA generally produces better class separation compared to PCA.
5. In the plots, LDA shows clearer boundaries between classes.
""")

# ==========================================================
# END
# ==========================================================
