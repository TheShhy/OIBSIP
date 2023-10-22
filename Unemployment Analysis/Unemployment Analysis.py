#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Unemployment.csv")
df.rename(columns={' Estimated Unemployment Rate (%)': 'Unemployment_Rate', 'Region': 'Region', ' Date': 'Date'}, inplace=True)
regional_avg_unemployment = df.groupby('Region')['Unemployment_Rate'].mean().reset_index()
df.columns = df.columns.str.strip()

print("Dataset Overview:")
print(df.head())
print(df.info())

missing_data = df.isnull().sum()
print("Missing Data:")
print(missing_data)

summary_stats = df.describe()
print("Summary Statistics:")
print(summary_stats)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Unemployment_Rate', hue='Region')
plt.title("Unemployment Rate Over Time by Region")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=regional_avg_unemployment, x='Region', y='Unemployment_Rate')
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.show()

unemployment_frequency = df['Frequency'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(unemployment_frequency, labels=unemployment_frequency.index, autopct='%1.1f%%', startangle=90)
plt.title("Unemployment Frequency Distribution")
plt.show()


# In[ ]:




