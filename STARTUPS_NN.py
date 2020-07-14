# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:12:26 2020

@author: SHRAVANI SRISAILAM
"""

import numpy as np
import pandas as pd

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
from sklearn.model_selection import train_test_split


########################## Neural Network for predicting continuous values ###############################

# Reading data 

startups = pd.read_csv("C:\\Users\\HP\\Desktop\\EXCELR\\ASSIGNMENTS\\NN\\50_Startups.csv")
startups.columns
startups = startups.rename(columns={'R&D Spend':'RD', 'Administration' : 'ADMIN', 'Marketing Spend' : 'MARKT'})
startups.head(5) # to get top 5 rows
type(startups)
startups.dtypes
startups['State'].unique() 

startups.loc[startups.State=="New York","State"] = 2
startups.loc[startups.State=="California","State"] = 1
startups.loc[startups.State=="Florida","State"] = 0
startups['State'].unique() 

startups.State.value_counts().plot(kind="bar")

################           ASSIGNING INPUT OUTPUT VARIABLES      ###################
# Let's divide into train and test set
from sklearn.model_selection import train_test_split
train,test = train_test_split(startups,test_size = 0.2,random_state=42)
trainX = train.drop(["State"],axis=1)
trainY = train["State"]
testX = test.drop(["State"],axis=1)
testY = test["State"]

trainX.shape, testX.shape
trainY.shape, testY.shape

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    # for the output layer we are not adding any activation function as 
    # the target variable is continuous variable 
    model.add(Dense(hidden_dim[-1]))
    # loss ---> loss function is means squared error to compare the output and estimated output
    # optimizer ---> adam
    # metrics ----> mean squared error - error for each epoch on entire data set 
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["mse"])
    return (model)

column_names = list(startups.columns)
predictors = column_names[0:4]
target = column_names[4]
startups[predictors] = np.asarray(startups[predictors]).astype(np.float32)
startups[target] = np.asarray(startups[target]).astype(np.float32)

first_model = prep_model([4,20,1])
first_model.fit(startups[predictors],startups[target],epochs=10)
pred_train = first_model.predict(startups[predictors])
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-startups[target])**2)) #86531

import matplotlib.pyplot as plt
plt.plot(pred_train,startups[target],"bo")
np.corrcoef(pred_train,startups[target]) # we 0.77 correlation 

# small picture -  ANN network and its layers 
from keras.utils import plot_model
plot_model(first_model,to_file="first_model.png")

#from ann_visualizer.visualize import ann_viz;

#ann_viz(first_model, title="My first neural network")
