# -------------- Classification: Logistic Regression --------------

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
# For L2 regularisation
from sklearn.linear_model import Ridge
# For calculating average generalisation error for each alpha, after L2 regularisation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load train and test data (x are attributes and y are true values)
x_train_df = pd.read_csv("C:/Users/s214680/OneDrive - Danmarks Tekniske Universitet/3. Semester/02450 Introduktion til machine learning og data mining/Project/project2_02450/2023-10-05_jennifer_data_preparation/independent_train.csv")
y_train_df = pd.read_csv("C:/Users/s214680/OneDrive - Danmarks Tekniske Universitet/3. Semester/02450 Introduktion til machine learning og data mining/Project/project2_02450/2023-10-05_jennifer_data_preparation/dependent_train.csv")
x_test_df = pd.read_csv("C:/Users/s214680/OneDrive - Danmarks Tekniske Universitet/3. Semester/02450 Introduktion til machine learning og data mining/Project/project2_02450/2023-10-05_jennifer_data_preparation/independent_test.csv")
y_test_df = pd.read_csv("C:/Users/s214680/OneDrive - Danmarks Tekniske Universitet/3. Semester/02450 Introduktion til machine learning og data mining/Project/project2_02450/2023-10-05_jennifer_data_preparation/dependent_test.csv")

# Flatten dependent variables (y) to 1D array
y_train = np.ravel(y_train_df)
y_test = np.ravel(y_test_df)

# Creating model
model = lm.LogisticRegression()

# Fitting training data to logistic regression model
model.fit(x_train_df, y_train)

# Predict
y_pred = model.predict(x_test_df)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f'Accuracy for Logistic Regression classification: {accuracy:.3f}')
print('Confusion Matrix: ')
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()












# With regularization parameter lambda

# Setting up lamda values and initialising an empty list of mean errors
lambda_values = np.logspace(-4, 4, 50)
conf_matrix_values = []
accuracy_values = []

for lambda_value in lambda_values:
    # Fit model with regularization parameter
    model = lm.LogisticRegression(C=lambda_value)
    model.fit(x_train_df, y_train)
    
    # Predict
    y_pred_value = model.predict(x_test_df)
    
    # Accuracy score
    accuracy_value = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy_value)



# L2 regularisation (Ridge linear regression).
# (Taken from Jennifers linear regression)

# Setting up lamda values and initialising an empty list of mean errors
lambda_values = np.logspace(-4, 4, 50)
mean_errors = []

# 10-Fold Cross-Validation for each lambda value:
for lambda_value in lambda_values:
    ridge = Ridge(alpha=lambda_value)
    scores = cross_val_score(ridge, 
                             x_train_df,
                             y_train,
                             cv=10,
                             scoring='neg_mean_squared_error')
    mean_errors.append(-np.mean(scores))

# Comparing the model coefficients to understand the effect of individual
# features on the predicted income.
coefs = []
for lambda_value in lambda_values:
    ridge = Ridge(alpha=lambda_value, fit_intercept=False)
    ridge.fit(x_train_df, y_train)
    coefs.append(ridge.coef_)

ax = plt.gca()

features = x_train_df.columns

for idx, feature in enumerate(features):
    ax.plot(lambda_values, np.array(np.squeeze(coefs))[:, idx], label=feature)

ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::1])  # reverse the axis
plt.xlabel("lambda")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.legend(loc="best", bbox_to_anchor=(1, 1), fontsize='small')  #Adding a legend
plt.axis("tight")
plt.show()



