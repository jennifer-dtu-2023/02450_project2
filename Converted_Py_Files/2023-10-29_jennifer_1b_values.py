#!/usr/bin/env python
# coding: utf-8

# ### Date: 2023-10-29
# ### Author: Jennifer Fortuny I Zhan
# ### Content: Project 2 Regression, Values for Linear Regression
# 
# In this analysis, we extend a basic linear regression model to include Ridge regularization for predicting income levels based on selected features.
# 
# We use a two-level cross-validation approach to optimize the regularization parameter lambda, aiming to minimize the error in each fold of the outerloop.
# 
# The process ends in identifying the best lambda values and corresponding test errors, which are important for evaluating the model.
# 
# Overall workflow:
# 1. Loading the pre-processed data, for dependent and independent variables of both training and testing sets.
# 2. Initialising the variables by setting up the number of folds for the outer and inner loops, i.e. K1 and K2, also initialised an empty array for potential lambda values.
# 3. Two-level cross-validation was implemented by:
#     - An outer loop which partitioned the data into a training set ad test sets.
#     - An inner loop which further partitioned the trainning set into inner training and validation sets.
#     - Training the model and calculating the error. By using Ridge regression models on the inner trainning set for each lambda, and calculating the validation error.
#     - Selecting the best lambda by selecting the lambda that minimised the average validation error for each fold in the outerloop.
#     - Testing the error calculation E^test, for each fold in the outer loop, using the best lambda.
# 4. Returning the best lambda values and E^test values for each fold, formated into 3 s.f.

# In[18]:


# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge


# In[5]:


# Load Data from CSV files
dependent_test_df = pd.read_csv('../2023-10-05_jennifer_data_preparation/dependent_test.csv')
dependent_train_df = pd.read_csv('../2023-10-05_jennifer_data_preparation/dependent_train.csv')
independent_test_df = pd.read_csv('../2023-10-05_jennifer_data_preparation/independent_test.csv')
independent_train_df = pd.read_csv('../2023-10-05_jennifer_data_preparation/independent_train.csv')


# In[6]:


# Converte DataFrames to NumPy arrays
X = independent_train_df.values
y = dependent_train_df.values


# In[19]:


# Initializing Variables:
# Number of folds for outer and inner loops
K1 = 10
K2 = 10

# Lambda values to test
lambdas = np.logspace(-4, 4, 50)

# Empty list to store best lambda values for each fold
best_lambdas = []
E_test_values = []


# In[20]:


# Function to calculate error
def calculate_error(y_true, y_pred):
    N_test = len(y_true)
    return (1 / N_test) * np.sum((y_true - y_pred)**2)

# Outer Loop
kf1 = KFold(n_splits=K1)
for train_index, test_index in kf1.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize a list to store the average errors for each lambda value
    avg_errors = np.zeros(len(lambdas))

    # Inner Loop
    kf2 = KFold(n_splits=K2)
    errors = []  # Reset errors list for each outerloop
    for inner_train_index, val_index in kf2.split(X_train):
        X_inner_train, X_val = X_train[inner_train_index], X_train[val_index]
        y_inner_train, y_val = y_train[inner_train_index], y_train[val_index]

        # Train Models and Calculate Errors
        for idx, l in enumerate(lambdas):
            model = Ridge(alpha=l)  # This adds the regularzation term.
            model.fit(X_inner_train, y_inner_train)
            y_pred = model.predict(X_val)
            error = calculate_error(y_val, y_pred)
            avg_errors[idx] += error  # This accumilates the erros for each lambda

    # Calculate the average errors for each lambda
    avg_errors /= K2
    
    # Select Best Lambda
    best_lambda = lambdas[np.argmin(avg_errors)]
    best_lambdas.append(best_lambda)

    # Train the best model on the entire training partition and calculate E^test
    best_model = Ridge(alpha=best_lambda)
    best_model.fit(X_train, y_train)
    y_test_pred = best_model.predict(X_test)
    E_test = calculate_error(y_test, y_test_pred)
    E_test_values.append(E_test)


# In[23]:


# Output the best lambda values and E^test values for each fold
print("Best Lambda values for each fold in Linear Regression Model:", [f'{x:.3f}' for x in best_lambdas])
print("E^test values for each fold:", [f'{x:.3f}' for x in E_test_values])

