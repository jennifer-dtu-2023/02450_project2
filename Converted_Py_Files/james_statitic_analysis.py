#!/usr/bin/env python
# coding: utf-8

# In[66]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency
from scipy.stats import binom


# In[62]:


# Load the data
independent_train_df = pd.read_csv('../2023-10-05_jennifer_data_preparation/independent_train.csv')
dependent_train_df = pd.read_csv('../2023-10-05_jennifer_data_preparation/dependent_train.csv')
dependent_train_df = dependent_train_df.values.ravel()


# <h1>3 models calculation for n_11, n_12, n_21, n_22 and the total<h1>

# In[75]:


def baseline_model(train, test):
    #Calaculate the most frequent appearance in the class (0, 1)
    most_freq_class = np.bincount(train.flatten()).argmax()
    predictions = np.full_like(test, most_freq_class)
    
    same_values_count = np.sum(predictions == test)
    
    total_elements = test.size
    
    return predictions, same_values_count, total_elements


# In[76]:


e = 0
t = np.array([1, 0, 0, 1, 0])
y = np.array([0, 0, 1, 1, 1])
u = np.array([1, 1, 1, 1, 1])

e += np.sum((t == u) & (u != y))
print(e)


# In[77]:


# Initialize variables
K = 10  # Number of folds
complexity_parameters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# K-Fold Cross-Validation
kf = KFold(n_splits=K)
y_pred_number_CT = 0
correct_number_CT = 0
y_pred_number_logist = 0
correct_number_logist = 0
y_pred_number_base = 0
correct_number_base = 0

n_11_LB = 0
n_12_LB = 0 
n_21_LB = 0
n_22_LB = 0
n_11_CL = 0
n_12_CL = 0 
n_21_CL = 0
n_22_CL = 0
n_11_CB = 0
n_12_CB = 0 
n_21_CB = 0
n_22_CB = 0



for train_index, test_index in tqdm(kf.split(independent_train_df)):
    X_train, X_test = independent_train_df.iloc[train_index], independent_train_df.iloc[test_index]
    y_train, y_test = dependent_train_df[train_index], dependent_train_df[test_index]

    
    #CT_algorithm
    best_correct = 0

    for cp in complexity_parameters:
        model = DecisionTreeClassifier(max_depth=cp)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        n_1 = sum(y_pred == y_test.ravel()) #james
        
        if n_1 > best_correct:    
            best_correct = n_1

    y_pred_number_CT += len(y_pred)
    correct_number_CT += best_correct
    
    
    
    
    #Logistic Regression
    param_grid = {'C': np.logspace(-4, 4, 50)}
    inner_model = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(inner_model, param_grid, cv=K, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Determine best lambda value
    best_lambda = grid_search.best_params_
    
    # Train model with best lambda value
    outer_model = LogisticRegression(max_iter=1000, **best_lambda)
    outer_model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred_outer = outer_model.predict(X_test)
    y_pred_number_logist += len(y_pred_outer)
    accuracy_outer = accuracy_score(y_test, y_pred_outer)
    correct_number_logist += int(accuracy_outer*len(y_pred_outer))
    
    
    
    
    #Baseline 
    predictions, currect_number, total_number = baseline_model(y_train, y_test)
    
    y_pred_number_base += total_number
    correct_number_base += currect_number




    #Compare models - Logist v.s Baseline
    n_11_LB += sum((y_pred_outer == y_test) & (predictions == y_test))
    n_12_LB += sum((y_pred_outer == y_test) & (predictions != y_test))
    n_21_LB += sum((y_pred_outer != y_test) & (predictions == y_test))
    n_22_LB += sum((y_pred_outer != y_test) & (predictions != y_test))
    
    #Compare models - CT v.s Logist
    n_11_CL += sum((y_pred == y_test) & (y_pred_outer == y_test))
    n_12_CL += sum((y_pred == y_test) & (y_pred_outer != y_test))
    n_21_CL += sum((y_pred != y_test) & (y_pred_outer == y_test))
    n_22_CL += sum((y_pred != y_test) & (y_pred_outer != y_test))
    
    #Compare models - CT v.s Baseline
    n_11_CB += sum((y_pred == y_test) & (predictions == y_test))
    n_12_CB += sum((y_pred == y_test) & (predictions != y_test))
    n_21_CB += sum((y_pred != y_test) & (predictions == y_test))
    n_22_CB += sum((y_pred != y_test) & (predictions != y_test))
    
    
print("n_1:", correct_number_CT)
print("n_1 percentage:", correct_number_CT / y_pred_number_CT)
print("n_1:", correct_number_logist)
print("n_1 percentage:", correct_number_logist / y_pred_number_logist)
print("n_1:", correct_number_base)
print("n_1 percentage:", correct_number_base / y_pred_number_base)

print("n_11_LB:", n_11_LB)
print("n_12_LB:", n_12_LB)
print("n_21_LB:", n_21_LB)
print("n_22_LB:", n_22_LB)
print("n_11_CL:", n_11_CL)
print("n_12_CL:", n_12_CL)
print("n_21_CL:", n_21_CL)
print("n_22_CL:", n_22_CL)
print("n_11_CB:", n_11_CB)
print("n_12_CB:", n_12_CB)
print("n_21_CB:", n_21_CB)
print("n_22_CB:", n_22_CB)


# <h1>McNermar Test<h1>

# In[85]:


def McNemar_t(n_11, n_12, n_21, n_22, n):
    
    confusion_matrix = [
        [n_11, n_12],
        [n_21, n_22]
    ]
    
    # McNemar's test
    result = mcnemar(confusion_matrix, exact=True)
    
    E_theta = (n_12 - n_21)/n
    
    # Calculate the confidence interval based on chi-squared distribution
    alpha = 0.05  # significance level 95% CI
    
    # Find the critical values for the confidence interval
    #theata_L_value = chi2_contingency(confusion_matrix, alpha / 2)[0]
    #theata_U_value = chi2_contingency(confusion_matrix, 1 - alpha / 2)[0]

    
    # Calculate the confidence interval
    #theata_L = 0.5 * (1 - np.sqrt(1 - theata_L_value / (n_11 + n_12)))
    #theata_U = 0.5 * (1 + np.sqrt(1 - theata_U_value / (n_11 + n_12)))
    
    alpha = 0.05  # 95% CI
    #Q = (n**2 * (n+1)*(E_theta + 1)*(1 - E_theta)) / (n*(n_12 + n_21) - (n_12 - n_21)**2)
    #f = (E_theta+1)/2 * (Q - 1)
    #g = (1 - E_theta)/2 * (Q - 1)
    n_CI = n_11 + n_12  
    p_CI = n_11 / n_CI
    
    theta_L = 2 * binom.ppf(alpha/2, n_CI, p_CI) / n_CI - 1
    theta_U = 2 * binom.ppf(1 - alpha/2, n_CI, p_CI) / n_CI - 1
    

    print("McNemar's test statistic:", result.statistic)
    print("p-value:", result.pvalue)
    print("E_theata:", E_theta)
    print(f"Confidence interval: ({theta_L}, {theta_U})")


# <h1>Compare Logistic Regression model and Baseline model<h1>

# In[86]:


t = McNemar_t(n_11_LB, n_12_LB, n_21_LB, n_22_LB, 256)
t


# <h1>Compare Classification Trees Values and Logistic Regression<h1>

# In[87]:


t = McNemar_t(n_11_CL, n_12_CL, n_21_CL, n_22_CL, 256)
t


# <h1>Compare Classification Trees Values and Baseline model<h1>

# In[88]:


t = McNemar_t(n_11_CB, n_12_CB, n_21_CB, n_22_CB, 256)
t

