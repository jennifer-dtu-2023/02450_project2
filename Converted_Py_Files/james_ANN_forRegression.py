#!/usr/bin/env python
# coding: utf-8

# <h1>Two-levels cross-validation for ANN model<h1>

# Follow our project description two-level cross-validation, 
# - Outer fold should be separated to train dataset and test dataset.
# - Inner fold training datasets should be divided to training and validation sets.
# - In inner fold, we should select the best hidden units to minimize the average validatation errors.
# - As we have already selected the best hidden units in each group, then compare each other by calculating the Generalization in outer fold.
# 

# In[54]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[55]:


import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold


# Load the dataframe from the csv files we stored

# In[56]:


train_x = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/independent_train.csv")
train_y = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/dependent_train.csv")
test_x = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/independent_test.csv")
test_y = pd.read_csv("/Users/luchengliang/02450_project2/2023-10-05_jennifer_data_preparation/dependent_test.csv")


# To clarify if we used the GPUs (Mac will choose mps and Windows will choose cuda for GPUs) or CPU.

# In[57]:


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# In[58]:


class ANN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(ANN_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.flatten = nn.Flatten()
        
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_hidden_layers)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 1), 
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x
    


# The weights needed to be reset in each fold. The following function will reset the parameters of the model. It could ensure the model is trained with the initailized randomly weights in order to avoid weight leakage.

# In[59]:


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# In[60]:


X = train_x
y = train_y

K1 = 10
K2 = 10
kfold_1 = KFold(n_splits=K1, shuffle=True)
kfold_2 = KFold(n_splits=K2, shuffle=True)

num_epochs = 30
criterion = nn.BCELoss()

input_dim = 25
hidden_dim_list = [1, 2, 3, 4, 5, 6, 7,  8, 9, 10]
hidden_layers = 1

torch.manual_seed(42)


# In[61]:


train_Tenx = torch.Tensor(X.to_numpy())
train_Teny = torch.Tensor(y.to_numpy())

dataset = TensorDataset(train_Tenx, train_Teny)


# In[62]:


outer_generalization_errors_list = []
best_hidden_units_list = []

for fold, (train_ids_out, test_ids_out) in enumerate(kfold_1.split(dataset)):
    
    print(f'Outer_FOLD: {fold + 1}')
    print('--------------------------------')
    
    unit_inner_generalization_errors_list = []
    
    for hidden_dim in hidden_dim_list:
        
        inner_generalization_errors_list = []
        
        for inner_fold, (inner_train_ids_out, inner_test_ids_out) in enumerate(kfold_2.split(train_ids_out)):
            
            print(f'Inner_FOLD: {inner_fold + 1}')
            print('--------------------------------')
            inner_fold_errors = []
            
            train_subsampler = SubsetRandomSampler(inner_train_ids_out)
            test_subsampler = SubsetRandomSampler(inner_test_ids_out)
            
            trainloader = DataLoader(dataset, batch_size = 10, sampler = train_subsampler)
            testloader = DataLoader(dataset, batch_size=10, sampler=test_subsampler)
            
            
            model = ANN_Model(input_dim, hidden_dim, hidden_layers - 1).to(device)
            model.apply(reset_weights)
            
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            
            for epoch in range(0, num_epochs):
                
                print(f'Starting epoch {epoch+1}')
                
                current_loss = 0.0
                model.train()
                for i, data in enumerate(trainloader, 0):
                    
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    optimizer.step()
                    
                    current_loss += loss.item()
                    if i % 100 == 99:
                        print('Loss after mini-batch %5d: %.3f' % (i+1, current_loss / 100))
                        current_loss = 0.0
            print('Inner fold training has finished.')
            
            print('Start testing')
            
            model.eval()
            with torch.no_grad():
                squared_loss = 0
                total_samples = 0
                for i , data in enumerate(testloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    squared_loss += ((outputs - labels)**2).sum().item()
                    total_samples += labels.size(0)
                mse = squared_loss / total_samples
                inner_fold_errors.append(mse)
            
            inner_generalization_error = sum(inner_fold_errors) / len(inner_fold_errors)
            inner_generalization_errors_list.append(inner_generalization_error)  
            print('--------------------------------')         
        unit_inner_generalization_error = sum(inner_generalization_errors_list) / len(inner_generalization_errors_list)
        unit_inner_generalization_errors_list.append(unit_inner_generalization_error)
    
    #Chose the best model by the number of hidden units
    
    outer_fold_errors = []
    
    train_subsampler_out = SubsetRandomSampler(train_ids_out)
    test_subsampler_out = SubsetRandomSampler(test_ids_out)
    
    trainloader_out = DataLoader(dataset, batch_size = 10, sampler = train_subsampler_out)
    testloader_out = DataLoader(dataset, batch_size=10, sampler=test_subsampler_out)
    
    best_hidden_unit = hidden_dim_list[np.argmin(unit_inner_generalization_errors_list)]
    
    best_model = ANN_Model(input_dim, best_hidden_unit, hidden_layers - 1).to(device)
    best_model.apply(reset_weights)
    
    optimizer = torch.optim.SGD(best_model.parameters(), lr=0.001)
    
    for epoch in range(0, num_epochs):
        
        print(f'Starting best model epoch {epoch+1}')
        
        current_loss = 0.0
        best_model.train()
        for i, data in enumerate(trainloader_out, 0):
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = best_model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            current_loss += loss.item()
            if i % 100 == 99:
                print('Loss after mini-batch %5d: %.3f' % (i+1, current_loss / 100))
                current_loss = 0.0
    print('Inner best model fold training has finished.')
    
    print('Start best model testing')
    
    best_model.eval()
    with torch.no_grad():
        squared_loss = 0
        total_samples = 0
        for i , data in enumerate(testloader_out, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs, 1)
            squared_loss += ((outputs - labels)**2).sum().item()
            total_samples += labels.size(0)
        mse = squared_loss / total_samples
        outer_fold_errors.append(mse)
        
    outer_generalization_error = sum(outer_fold_errors) / len(outer_fold_errors)
    outer_generalization_errors_list.append(outer_generalization_error)
    best_hidden_units_list.append(best_hidden_unit) 
    print('--------------------------------')
                
        


# In[64]:


print("Best hidden units for each outer-fold in ANN Model:", [f'{x}' for x in best_hidden_units_list])
print("E^test values for each fold:", [f'{x:.3f}' for x in outer_generalization_errors_list])

