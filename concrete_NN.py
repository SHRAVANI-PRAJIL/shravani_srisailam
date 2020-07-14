# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:08:15 2020

@author: HP
"""

import numpy as np
import pandas as pd

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
from sklearn.model_selection import train_test_split


########################## Neural Network for predicting continuous values ###############################

# Reading data 
Concrete = pd.read_csv("D:\\ExcelR\\ASSIGNMENTS\\NN\\concrete.csv")
Concrete.head()
Concrete.shape
Concrete.describe().transpose()
Concrete.corr()
Concrete.info()
Concrete.isnull().sum()
Concrete.columns
###  look for any outliers in the data
import seaborn as sns
#sns.boxplot(x=Concrete['cement'])
sns.boxplot(x=Concrete['slag'])
#sns.boxplot(x=Concrete['ash'])
sns.boxplot(x=Concrete['water'])
sns.boxplot(x=Concrete['superplastic'])
#sns.boxplot(x=Concrete['coarseagg'])
sns.boxplot(x=Concrete['fineagg'])
sns.boxplot(x=Concrete['age'])

#defining the new data set after removing outliers
Q1 = Concrete.quantile(0.25)
Q3 = Concrete.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df_out = Concrete[~((Concrete < (Q1 - 1.5 * IQR)) |(Concrete > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_out.shape)
# NEW DATASET
Concrete = df_out
Concrete.shape

######### model building  
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

column_names = list(Concrete.columns)
predictors = column_names[0:8]
target = column_names[8]

first_model = prep_model([8,50,1])
first_model.fit(np.array(Concrete[predictors]),np.array(Concrete[target]),epochs=10)
pred_train = first_model.predict(np.array(Concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-Concrete[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,Concrete[target],"bo")
np.corrcoef(pred_train,Concrete[target]) # we got high correlation 

# small picture -  ANN network and its layers 
from keras.utils import plot_model
plot_model(first_model,to_file="first_model.png")


#from ann_visualizer.visualize import ann_viz;

#ann_viz(first_model, title="My first neural network")
