# -------------- Regression: Baseline --------------

# The baseline model computes the mean of train_y
# data and uses this value to predict test_y data.

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

# Load train and test data (x are attributes and y are true values)
x_train = pd.read_csv("C:/Users/s214680/OneDrive - Danmarks Tekniske Universitet/3. Semester/02450 Introduktion til machine learning og data mining/Project/project2_02450/2023-10-05_jennifer_data_preparation/independent_train.csv")
y_train = pd.read_csv("C:/Users/s214680/OneDrive - Danmarks Tekniske Universitet/3. Semester/02450 Introduktion til machine learning og data mining/Project/project2_02450/2023-10-05_jennifer_data_preparation/dependent_train.csv")
x_test = pd.read_csv("C:/Users/s214680/OneDrive - Danmarks Tekniske Universitet/3. Semester/02450 Introduktion til machine learning og data mining/Project/project2_02450/2023-10-05_jennifer_data_preparation/independent_test.csv")
y_test = pd.read_csv("C:/Users/s214680/OneDrive - Danmarks Tekniske Universitet/3. Semester/02450 Introduktion til machine learning og data mining/Project/project2_02450/2023-10-05_jennifer_data_preparation/dependent_test.csv")
# "income" is binary-coded (0 or 1)

# Compute mean of train_y
temp = y_train.mean()
mean_y_train = temp[0]

# Create linear regression model
baseline_model = lm.LinearRegression()

# Train model by fitting to mean of y
baseline_model.fit(x_train, np.full_like(y_train, mean_y_train))

# Predict y on the test data
y_pred = baseline_model.predict(x_test)

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


# plt.figure(figsize=(10, 5))
# #plt.plot(x_train['age'], y_train, 'o', label='Training data')
# plt.plot(x_test['age'], y_test, 'o', label='Test data')
# plt.plot(x_test['age'], y_pred, '-', label='Regression fit (model)')
# plt.xlabel('x', fontsize=20); plt.ylabel('y', fontsize=20)
# plt.xticks(fontsize=16); plt.yticks(fontsize=16);
# plt.title("Baseline Linear Regression Model", fontsize=24)
# plt.legend(fontsize=20)
# plt.show()

# ctrl + 1 to comment block


