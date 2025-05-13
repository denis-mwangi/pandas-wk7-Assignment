# Task 1: Load and Explore the Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Display first few rows
print(df.head())

# Explore structure
print(df.info())
print(df.isnull().sum())  # Check for missing values

# Task 2
# Descriptive statistics
print(df.describe())

# Group by species and compute mean of each numerical column
grouped = df.groupby("target").mean()
print(grouped)

# Replace numerical target with species names for clarity
df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))

# View means grouped by species
print(df.groupby("species").mean())

#Task 3
# Line chart - sepal length trend (simulating a time-like series)
plt.figure(figsize=(10, 5))
plt.plot(df['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length Trend')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# Bar chart - average petal length per species
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# Histogram - distribution of sepal width
plt.figure(figsize=(8, 6))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot - sepal length vs petal length
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()
