# ==========================================================
# SPAM EMAIL FILTERING USING NAIVE BAYES CLASSIFIER
# ==========================================================

# ==========================================================
# STEP 1: IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ==========================================================
# STEP 2: LOAD DATASET
# ==========================================================

# Download dataset from:
# https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

# Replace with your file path
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("Dataset Head:")
print(df.head())

# ==========================================================
# STEP 3: DATA PREPROCESSING
# ==========================================================

# Convert labels into numeric values
# ham = 0, spam = 1

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nDataset Information:")
print(df.info())

print("\nChecking Missing Values:")
print(df.isnull().sum())

# ==========================================================
# STEP 4: SPLIT FEATURES AND TARGET
# ==========================================================

X = df['message']
y = df['label']

# ==========================================================
# STEP 5: CONVERT TEXT INTO NUMERICAL FORM
# ==========================================================

vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# ==========================================================
# STEP 6: TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# STEP 7: APPLY NAIVE BAYES CLASSIFIER
# ==========================================================

model = MultinomialNB()

model.fit(X_train, y_train)

# ==========================================================
# STEP 8: MAKE PREDICTIONS
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# STEP 9: EVALUATE MODEL
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================================
# STEP 10: TEST WITH CUSTOM EMAIL
# ==========================================================

sample_email = [
    "Congratulations! You have won a free iPhone. Click now to claim prize."
]

sample_vector = vectorizer.transform(sample_email)

prediction = model.predict(sample_vector)

print("\n===================================")
print("CUSTOM EMAIL PREDICTION")
print("===================================")

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")

# ==========================================================
# END
# ==========================================================
