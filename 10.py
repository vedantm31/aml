# ==========================================================
# MARKET BASKET ANALYSIS USING APRIORI ALGORITHM
# ==========================================================

# ==========================================================
# STEP 1: IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# ==========================================================
# STEP 2: LOAD DATASET
# ==========================================================

# Dataset Source:
# https://github.com/kkrusere/Market-Basket-Analysis-on-the-Online-Retail-Data

# Replace with your file path
df = pd.read_csv("OnlineRetail.csv", encoding='latin1')

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
# STEP 4: HANDLE MISSING VALUES
# ==========================================================

# Remove rows with missing values
df.dropna(inplace=True)

print("\nDataset Shape After Removing Missing Values:")
print(df.shape)

# ==========================================================
# STEP 5: DATA CLEANING
# ==========================================================

# Remove cancelled transactions
df = df[~df['InvoiceNo'].astype(str).str.contains('C')]

# Remove negative or zero quantity
df = df[df['Quantity'] > 0]

# ==========================================================
# STEP 6: TOP SELLING PRODUCTS
# ==========================================================

top_products = df['Description'].value_counts().head(10)

print("\nTop Selling Products:")
print(top_products)

# Plot Top Products
plt.figure(figsize=(10,6))

top_products.plot(kind='bar')

plt.title("Top Selling Products")

plt.xlabel("Products")

plt.ylabel("Count")

plt.show()

# ==========================================================
# STEP 7: CREATE MARKET BASKET
# ==========================================================

basket = (
    df.groupby(['InvoiceNo', 'Description'])['Quantity']
    .sum()
    .unstack()
    .fillna(0)
)

# Convert quantities into binary values
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

print("\nBasket Dataset:")
print(basket.head())

# ==========================================================
# STEP 8: APPLY APRIORI ALGORITHM
# ==========================================================

frequent_itemsets = apriori(
    basket,
    min_support=0.02,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets.head())

# ==========================================================
# STEP 9: GENERATE ASSOCIATION RULES
# ==========================================================

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
)

# Sort by lift
rules = rules.sort_values(by='lift', ascending=False)

print("\nAssociation Rules:")
print(rules.head(10))

# ==========================================================
# STEP 10: DISPLAY IMPORTANT RULES
# ==========================================================

print("\n===================================")
print("TOP ASSOCIATION RULES")
print("===================================")

for index, row in rules.head(10).iterrows():

    print("\nRule:")
    print(list(row['antecedents']), "=>", list(row['consequents']))

    print("Support:", row['support'])

    print("Confidence:", row['confidence'])

    print("Lift:", row['lift'])

# ==========================================================
# STEP 11: VISUALIZE SUPPORT VS CONFIDENCE
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=rules['support'],
    y=rules['confidence'],
    size=rules['lift'],
    legend=False
)

plt.title("Support vs Confidence")

plt.xlabel("Support")

plt.ylabel("Confidence")

plt.show()

# ==========================================================
# STEP 12: CONCLUSION
# ==========================================================

print("\n===================================")
print("CONCLUSION")
print("===================================")

print("""
1. Apriori algorithm was used to identify frequent itemsets.
2. Association rules help identify products that are frequently bought together.
3. Businesses can use these rules for:
   - Product recommendations
   - Cross-selling
   - Store layout optimization
   - Customer segmentation
4. High lift values indicate strong associations between products.
""")

# ==========================================================
# END
# ==========================================================
