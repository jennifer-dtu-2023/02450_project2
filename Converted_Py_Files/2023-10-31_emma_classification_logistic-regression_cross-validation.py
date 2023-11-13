#!/usr/bin/env python
# coding: utf-8

# # Two-level cross-validation on the logistic regression model

# ### Date: 2023-10-31
# ### Author: Emma Louise Blair (s214680)

# The purpose of the two-level cross-validation on the logistic regression model is to analyze how well the model is able to categorize whether an individual has a yearly income below or above 50K/y dollars. We used a 10-fold cross-validation and hyperparameter tuning to optimize the model by minimizing the test error value for each fold. For a logistic regression model the hyperparameter in question is the L2 regularization parameter $\lambda$. We use ```GridSearchCV()``` to find the best $\lambda$-value based on the accuracy score for each $\lambda$-value.

# In[2]:


# Imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score


# In[3]:


# Load data
X_df = pd.read_csv("../2023-10-05_jennifer_data_preparation/independent_train.csv")
y_df = pd.read_csv("../2023-10-05_jennifer_data_preparation/dependent_train.csv")


# In[4]:


# Flatten dependent variables (y) to 1D array
y_df = y_df.values.ravel()


# In[5]:


# Initialize variables and lists
K = 10
kfold = KFold(n_splits=K)

best_lambdas = []
E_test_values = []


# In[10]:


# Outer loop
for train_idx, test_idx in kfold.split(X_df, y_df):
    X_train_outer, X_test_outer = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train_outer, y_test_outer = y_df[train_idx], y_df[test_idx]
    
    param_grid = {'C': np.logspace(-4, 4, 50)}
    inner_model = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(inner_model, param_grid, cv=K, scoring='accuracy')
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Determine best lambda value
    best_lambda = grid_search.best_params_
    
    # Train model with best lambda value
    outer_model = LogisticRegression(max_iter=1000, **best_lambda)
    outer_model.fit(X_train_outer, y_train_outer)
    
    # Calculate accuracy
    y_pred_outer = outer_model.predict(X_test_outer)
    accuracy_outer = accuracy_score(y_test_outer, y_pred_outer)
    
    # Store best lambda value and corresponding E^test value (E^test = 1 - accuracy)
    best_lambdas.append(best_lambda)
    E_test_values.append(1 - accuracy_outer)


# In[7]:


print("Best lambda value for each fold in the model:", [f'{x["C"]:.3f}' for x in best_lambdas])
print("E^test value for each fold in the model:", [f'{x:.3f}' for x in E_test_values])

