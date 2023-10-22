#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("Sales.csv")

# Extract features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_predictions = linear_reg.predict(X_test)

# Visualize future sales predictions for TV
plt.figure(figsize=(10, 6))
plt.scatter(X_test['TV'], y_test, color='blue', label='Actual Sales')
plt.scatter(X_test['TV'], linear_reg_predictions, color='red', label='Predicted Sales')
plt.title("Future Sales Predictions for TV Advertising")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Visualize future sales predictions for Radio
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Radio'], y_test, color='blue', label='Actual Sales')
plt.scatter(X_test['Radio'], linear_reg_predictions, color='red', label='Predicted Sales')
plt.title("Future Sales Predictions for Radio Advertising")
plt.xlabel("Radio Advertising Budget")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Visualize future sales predictions for Newspaper
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Newspaper'], y_test, color='blue', label='Actual Sales')
plt.scatter(X_test['Newspaper'], linear_reg_predictions, color='red', label='Predicted Sales')
plt.title("Future Sales Predictions for Newspaper Advertising")
plt.xlabel("Newspaper Advertising Budget")
plt.ylabel("Sales")
plt.legend()
plt.show()


# In[ ]:




