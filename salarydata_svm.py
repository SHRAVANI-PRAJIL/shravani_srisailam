# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:32:57 2020

@author: SHRAVANI PRAJIL
"""


import pandas as pd 
import numpy as np 
import seaborn as sns
# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
# for one hot encoding with feature-engine
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
##################              Reading the Salary Data        ####################

salary_train = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\NB\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\NB\\SalaryData_Test.csv")
salary_train.describe()
salary_train.dtypes
salary_train['Salary'] = salary_train['Salary'].apply(lambda x: '0' if x==' <=50K' else '1')
salary_test['Salary'] = salary_test['Salary'].apply(lambda x: '0' if x==' <=50K' else '1')
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
salary_train.describe()
salary_test.describe()

#train[numerical]
k=salary_train[string_columns]

# let's have a look at how many labels in train data for  each variable has
for col in k.columns:
    print(col, ': ', len(salary_train[col].unique()), ' labels')

#to find most frequent labels in native categorical varible 
# let's find the top 10 most frequent categories for the variable 'Neighborhood'

salary_train['native'].value_counts().sort_values(ascending=False).head(40)
ohe_enc = OneHotCategoricalEncoder(
    top_categories=14,  # you can change this value to select more or less variables
    # we can select which variables to encode
    variables=['workclass','education','maritalstatus','occupation','relationship','race','sex','native'],
    drop_last=False)

ohe_enc.fit(salary_train)
ohe_enc.fit(salary_test)

# in the encoder dict we can observe each of the top categories
# selected for each of the variables

ohe_enc.encoder_dict_

# this is the list of variables that the encoder will transform

ohe_enc.variables

salary_train = ohe_enc.transform(salary_train)
salary_test = ohe_enc.transform(salary_test)

# let's explore the result
salary_train.head()
salary_test.head()

#######                 sns.pairplot(data=salary_traiN)           ##################
sns.boxplot(x="Salary",y="age",data=salary_train,palette = "hls")
############              sns.pairplot(data=salary_test)          #################
sns.boxplot(x="Salary",y="age",data=salary_test,palette = "hls")
################           ASSIGNING INPUT OUTPUT VARIABLES      ###################

colnames = salary_train.columns
train_X = salary_train[colnames[0:13]]
train_y = salary_train[colnames[13]]
test_X  = salary_test[colnames[0:13]]
test_y  = salary_test[colnames[13]]
salary_train.info
salary_test.info

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

################            Create SVM classification object                    ####################
#                  'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'                       #########
from sklearn.svm import SVC
# # Initialize SVM classifier for kernel = linear

help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 67.19%

################                   dealing with imbalanced data             ###################
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(train_X,np.array( train_y).ravel())

###  building svm model after applying SMOTE 
model_linear.fit(X_train_res,y_train_res)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 65.79%

# # Initialize SVM classifier for kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
#           POLY MODEL ACCURACY = 
np.mean(pred_test_poly==test_y) # Accuracy = 87.30%


## # Initialize SVM classifier for kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
#           RBF MODEL ACCURACY = 
np.mean(pred_test_rbf==test_y) # Accuracy = 96.66%


# # # Initialize SVM classifier for kernel = sigmoid
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_X,train_y)
pred_test_sig = model_sig.predict(test_X)
#           sigmoid MODEL ACCURACY = 
np.mean(pred_test_sig==test_y) # Accuracy = 62.76
