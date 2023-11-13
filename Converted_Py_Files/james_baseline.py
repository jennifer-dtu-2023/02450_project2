#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[45]:


train_x = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/independent_train.csv")
train_y = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/dependent_train.csv")
test_x = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/independent_test.csv")
test_y = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/dependent_test.csv")
print(train_x.shape)
train_x.head(5)


# In[46]:


print(train_y.shape)
train_y.head(5)


# In[47]:


print(test_x.shape)
test_x.head(5)


# In[48]:


print(test_y.shape)
test_y.head(5)


# In[49]:


#Calaculate the most frequent appearance in the class (0, 1)
most_freq_class = np.bincount(train_y.to_numpy().flatten()).argmax()
most_freq_class


# In[50]:


#As the baseline will be a model which compute the largest class on the training data, 
# and predict everything in the test-data as belonging to that class
predictions = np.full_like(test_y, most_freq_class)
predictions.shape


# In[51]:


confusion_matrix = confusion_matrix(test_y, predictions)
accuracy = accuracy_score(test_y, predictions)

# Print results
print(f'Accuracy for Baseline model classification: {accuracy:.3f}')
print('Confusion Matrix: ')
print(confusion_matrix)


# In[54]:


group_names = ['True Negative', 'False Positive', 'False Negetive', 'True Positive']
group_counts = ["{0:0.0f}".format(value) for value in
                confusion_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     confusion_matrix.flatten()/np.sum(confusion_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap="crest")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

