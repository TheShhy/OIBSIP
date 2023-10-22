#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Iris.csv')

X = dataset.drop(columns=['Species'])
y = dataset['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression(max_iter=1000)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(4, 4))
plt.text(0.5, 0.5, f'Accuracy: {accuracy * 100:.2f}%', fontsize=18, ha='center', va='center')
plt.axis('off')
plt.title('Accuracy')
plt.show()


# In[ ]:




