import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load dataset (California Housing dataset as an example)
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target variable
df['Price'] = data.target  

# Display the first few rows
#print(df.head())

# Check for missing values
#print(df.isnull().sum())

# Display summary statistics
#print(df.describe())

# Plot histogram of the target variable
'''
plt.figure(figsize=(8, 4))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.show()
'''

# Boxplots to check for outliers
'''
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient='h')
plt.title("Feature Distributions")
plt.show()
'''

# Pairplot to check for relationships
# (MedInc = Median Income, which usually correlates with house price.)
'''
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'Price']])
plt.show()
'''

# Correlation Heatmap <-- This is the most helpful for students.
# (This helps students identify the most relevant predictors for regression.)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Scatter Plots to Examine Key Trends
# This step helps students grasp if a linear regression model is suitable (e.g., is there a linear trend? Are there outliers?).
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['MedInc'], y=df['Price'])
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("House Price vs Median Income")
plt.show()

