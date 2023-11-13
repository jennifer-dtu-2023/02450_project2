#!/usr/bin/env python
# coding: utf-8

# <h1>Two-level cross validation for baseline<h1>

# In this section, we used our baseline model to predict income if they are surpassed 50k per month. We use two-level cross validation to see if we could improve our prediction.

# In[4]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold


# In[10]:


train_x = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/independent_train.csv")
train_y = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/dependent_train.csv")

X = train_x.values
y = train_y.values

K1 = 10
K2 = 10
kfold_1 = KFold(n_splits=K1, shuffle=True)
kfold_2 = KFold(n_splits=K2, shuffle=True)


# In[16]:


def baseline_model(train, test):
    #Calaculate the most frequent appearance in the class (0, 1)
    most_freq_class = np.bincount(train.flatten()).argmax()
    predictions = np.full_like(test, most_freq_class)
    
    diff_values_count = np.sum(predictions != test)
    
    total_elements = test.size
    
    E_val = diff_values_count / total_elements
    
    return E_val


# In[18]:


outer_Generalization_errors = []

for train_ids, test_ids in kfold_1.split(X):
    X_train, X_test = X[train_ids], X[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]
    
    inner_Generalization_errors = []
    
    for inner_train_ids, inner_test_ids in kfold_2.split(X_train):
        inner_X_train, inner_X_val = X[inner_train_ids], X[inner_test_ids]
        inner_y_train, inner_y_val = y[inner_train_ids], y[inner_test_ids]
        
        inner_E_val = baseline_model(inner_y_train, inner_y_val)
        
        inner_Generalization_errors.append(inner_E_val)
    
    outer_E_val = baseline_model(y_train, y_test)
    
    outer_Generalization_errors.append(outer_E_val)
    


# In[20]:


print("E^test values for each inner-fold:", [f'{x:.3f}' for x in inner_Generalization_errors])
print("E^test values for each outer-fold:", [f'{x:.3f}' for x in outer_Generalization_errors])


# As we didn't modify or try the other methods for our model, so even we used this optimization way. The model still could not be improved and lowered its generalization errors.
