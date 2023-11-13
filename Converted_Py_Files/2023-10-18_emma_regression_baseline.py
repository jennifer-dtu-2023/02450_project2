#!/usr/bin/env python
# coding: utf-8

# # Project 2 - Baseline Regression
# ### Date: 2023-10-18
# ### Author: Emma Louise Blair

# In[11]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm


# In[12]:


# Load train and test data (x are attributes and y are true values)
x_train = pd.read_csv("../2023-10-05_jennifer_data_preparation/independent_train.csv")
y_train = pd.read_csv("../2023-10-05_jennifer_data_preparation/dependent_train.csv")
x_test = pd.read_csv("../2023-10-05_jennifer_data_preparation/independent_test.csv")
y_test = pd.read_csv("../2023-10-05_jennifer_data_preparation/dependent_test.csv")


# ```income``` is binary-coded.

# The baseline model computes the mean of y_train data and uses this value to predict y_test data.

# In[13]:


# Compute mean of train_y
temp = y_train.mean()
mean_y_train = temp[0]


# We use ```LinearRegression()``` from ```sklearn.linear_model``` to model the baseline.

# In[14]:


# Create linear regression model
baseline_model = lm.LinearRegression()


# We train the model by fitting the x training data to the mean of the y training data.

# In[15]:


# Train model by fitting to mean of y
baseline_model.fit(x_train, np.full_like(y_train, mean_y_train))


# We can then use the baseline model to predict y values based on the x test data.

# In[16]:


# Predict y on the test data
y_pred = baseline_model.predict(x_test)


# We will now plot ```y_test``` as a function of ```x_test``` and ```y_pred``` as a function of ```x_test``` for each of the attributes.

# In[17]:


# --- Plot original data and the model output
plt.figure(figsize=(10, 8))

# Subplot 1 - Feature: age
plt.subplot(2, 2, 1)
plt.plot(x_test['age'], y_test, 'o')
plt.plot(x_test['age'], y_pred, '-')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(['Test data', 'Regression fit (model)'])
plt.title("Feature: age")

# Subplot 2 - Feature: education-num
plt.subplot(2, 2, 2)
plt.plot(x_test['education-num'], y_test, 'o')
plt.plot(x_test['education-num'], y_pred, '-')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(['Test data', 'Regression fit (model)'])
plt.title("Feature: education number")

# Subplot 3 - Feature: hours-per-week
plt.subplot(2, 2, 3)
plt.plot(x_test['hours-per-week'], y_test, 'o')
plt.plot(x_test['hours-per-week'], y_pred, '-')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(['Test data', 'Regression fit (model)'])
plt.title("Feature: hours-per-week")

# Subplot 4 - Feature: workclass_ ?
plt.subplot(2, 2, 4)
plt.plot(x_test['workclass_ ?'], y_test, 'o')
plt.plot(x_test['workclass_ ?'], y_pred, '-')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(['Test data', 'Regression fit (model)'])
plt.title("Feature: workclass_ ?")

plt.tight_layout()
plt.show()

