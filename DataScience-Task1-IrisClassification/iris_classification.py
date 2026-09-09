# TASK 1 - Iris Flower Classification
# By Chandani Mishra - OIBSIP Data Science

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

print("Shape:", df.shape)
print(df.head())
print("Null values:", df.isnull().sum())
print(df.describe())

# EDA - Pairplot
sns.pairplot(df, hue='species_name')
plt.show()

# EDA - Boxplot
for col in iris.feature_names:
    plt.figure()
    sns.boxplot(x='species_name', y=col, data=df)
    plt.title(col)
    plt.show()

# Train Test Split
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model - Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Second model - Random Forest for better accuracy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
