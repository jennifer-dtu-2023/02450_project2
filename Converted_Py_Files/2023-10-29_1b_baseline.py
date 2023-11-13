#!/usr/bin/env python
# coding: utf-8

# # Two-level cross-validation on the baseline model

# ### Date: 2023-10-29
# ### Author: Emma Louise Blair (s214680)

# In[15]:


# Load train data
x_train = pd.read_csv("../2023-10-05_jennifer_data_preparation/independent_train.csv")
y_train = pd.read_csv("../2023-10-05_jennifer_data_preparation/dependent_train.csv")


# In[16]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms


# The baseline model computes the mean of ```y_train```  data and uses this value to predict ```y_test``` data.

# We use Algorithm 6: Two-level cross-validation on p. 177 in the book. The below code is partly taken from ex_6_2_1.py.

# In[17]:


# Convert from Dataframes to NumPy arrays
X = x_train.values
y = y_train.values


# In[18]:


# No. of folds in outer and inner cross-validation loop, respectively
K1 = 10
K2 = 10


# In[19]:


# Initialize E^test array
E_test_values = []


# In[20]:


# Function to calculate error
def calculate_error(y_true, y_pred):
    N_test = len(y_true)
    return (1 / N_test) * np.sum((y_true - y_pred)**2)


# In[21]:


# Outer loop
kfold1 = ms.KFold(n_splits=K1)
for train_index1, test_index1 in kfold1.split(X):
    X_train, X_test = X[train_index1], X[test_index1]
    y_train, y_test = y[train_index1], y[test_index1]
    
    # Inner loop
    kfold2 = ms.KFold(n_splits=K2)
    for train_index2, test_index2 in kfold2.split(X_train):
        X_train_inner, X_val = X_train[train_index2], X_train[test_index2]
        y_train_inner, y_val = y_train[train_index2], y_train[test_index2]
    
    # Train model and calculate E^test
    baseline_model = lm.LinearRegression()
    baseline_model.fit(X_train, y_train)
    y_pred = baseline_model.predict(X_test)
    E_test = calculate_error(y_test, y_pred)
    E_test_values.append(E_test)


# In[22]:


# Print E^test values for each fold
print("E^test values for each fold:", [f'{x:.3f}' for x in E_test_values])

