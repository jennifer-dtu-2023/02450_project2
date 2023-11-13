#!/usr/bin/env python
# coding: utf-8

# ### Date: 2023-10-29
# ### Author: Jennifer Fortuny I Zhan
# ### Content: Project 2 Classification, Classification Trees Values
# 
# In this analysis, we employ a Decision Tree Classifier to categorize individuals based on their income levels, either above or below $50K.
# 
# We used 10-fold cross-validation and hyperparameter tuning to optimize the model, focusing on minimizing the test error rate.
# 
# Workflow Overview:
# 1. Loaded the pre-processed training data.
# 2. Initialized the 10-fold cross-validation and a set of complexity parameters
# 3. Performed a 10-fold cross-validation by
#     - Training the Classification Tree using different complexity parameters for each fold.
#     - Computing the test error rate for each fold an each complexity parameter.
# 4. Recorded the best complexity parameter and test error rate for each fold.
# 5. Printed the best complexity values and test error rates for each fold

# In[23]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# Load the data
independent_train_df = pd.read_csv('../2023-10-05_jennifer_data_preparation/independent_train.csv')
dependent_train_df = pd.read_csv('../2023-10-05_jennifer_data_preparation/dependent_train.csv')

# Initialize variables
K = 10  # Number of folds
complexity_parameters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

best_complexity_values = []
E_test_values = []

# K-Fold Cross-Validation
kf = KFold(n_splits=K)

for train_index, test_index in kf.split(independent_train_df):
    X_train, X_test = independent_train_df.iloc[train_index], independent_train_df.iloc[test_index]
    y_train, y_test = dependent_train_df.iloc[train_index], dependent_train_df.iloc[test_index]

    best_error = float('inf')
    best_complexity = None

    for cp in complexity_parameters:
        model = DecisionTreeClassifier(max_depth=cp)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = np.mean(y_pred != y_test.to_numpy().ravel())

        if error < best_error:
            best_error = error
            best_complexity = cp

    best_complexity_values.append(best_complexity)
    E_test_values.append(best_error)

print(f'Best complexity values for each fold: {best_complexity_values}')
print(f'Test error rate for each fold: {["{:.3f}".format(x) for x in E_test_values]}')


# From the restuls, 3 is the most frequently appearing best complexity value, making it a candidate for further refinement.
# The test error rates ranged from 0.077 to 0.280, this suggests that the model has some predictive power with areas for further improvement.
