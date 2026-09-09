# TASK 3 Sales Prediction - By Chandani Mishra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Load data (download Advertising.csv)
df = pd.read_csv('Advertising.csv')
print(df.head(), df.shape)
print(df.isnull().sum())

sns.pairplot(df)
plt.show()

for col in ['TV','Radio','Newspaper']:
    plt.scatter(df[col], df['Sales'])
    plt.xlabel(col); plt.ylabel('Sales'); plt.show()

X = df[['TV','Radio','Newspaper']]
y = df['Sales']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
print("R2:", r2_score(y_test,pred))
print("MAE:", mean_absolute_error(y_test,pred))
print("Coef:", lr.coef_)
print("TV has max impact on Sales")
