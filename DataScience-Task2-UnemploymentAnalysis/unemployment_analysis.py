# TASK 2 - Unemployment Analysis
# By Chandani Mishra - OIBSIP Data Science

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# If you have CSV, change path, else sample logic below will work for demo
try:
    df = pd.read_csv('Unemployment in India.csv')
except:
    print("CSV not found, using sample structure for GitHub")
    df = pd.read_csv('https://raw.githubusercontent.com/Chandanimishra/OIBSIP/main/sample.csv')
    
# Basic info
print(df.shape)
print(df.head())
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Region wise average
avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=avg.head(10).index, y=avg.head(10).values)
plt.xticks(rotation=45)
plt.title('Top 10 States Highest Unemployment')
plt.show()

# Time trend
plt.figure(figsize=(12,5))
for state in df['Region'].unique()[:3]:
    data = df[df['Region']==state]
    plt.plot(data['Date'], data['Estimated Unemployment Rate (%)'], label=state)
plt.legend()
plt.title('Unemployment Trend')
plt.show()

# Correlation heatmap
sns.heatmap(df[['Estimated Unemployment Rate (%)','Estimated Employed','Estimated Labour Participation Rate (%)']].corr(), annot=True)
plt.show()

print("Conclusion: Unemployment increased during Covid lockdown")
