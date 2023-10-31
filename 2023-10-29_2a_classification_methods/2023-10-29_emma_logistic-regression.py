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


