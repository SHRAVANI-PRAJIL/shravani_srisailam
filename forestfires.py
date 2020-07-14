# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:16:18 2020

@author: SHRAVANI PRAJIL
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:40:50 2020

@author: SHRAVANI PRAJIL
"""


##################              importing packages             ####################
import pandas as pd 
import numpy as np 
import seaborn as sns
##################              Reading the Salary Data        ####################

salary_train = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\NB\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\NB\\SalaryData_Test.csv")
salary_train.describe()
salary_train.dtypes
salary_train['Salary'] = salary_train['Salary'].apply(lambda x: (-1) if x==' <=50K' else 1)
salary_test['Salary'] = salary_test['Salary'].apply(lambda x: (-1) if x==' <=50K' else 1)
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
salary_train.describe()
salary_test.describe()

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])
    
#######                 sns.pairplot(data=salary_traiN)           ##################
sns.boxplot(x="Salary",y="age",data=salary_train,palette = "hls")
sns.boxplot(x="Salary",y="education",data=salary_train,palette = "hls")
############              sns.pairplot(data=salary_test)          #################
sns.boxplot(x="Salary",y="age",data=salary_test,palette = "hls")
sns.boxplot(x="Salary",y="education",data=salary_test,palette = "hls")
 
################           ASSIGNING INPUT OUTPUT VARIABLES      ###################
colnames = salary_train.columns
train_X = salary_train[colnames[0:13]]
train_y = salary_train[colnames[13]]
test_X  = salary_test[colnames[0:13]]
test_y  = salary_test[colnames[13]]

################            Create SVM classification object                    ####################
#                  'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'                       #########
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
#           LINEAR MODEL ACCURACY = 
np.mean(pred_test_linear==test_y) # Accuracy = 86.116
# Get support vector indices
support_vector_indices = clf.support_
print(support_vector_indices)


# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
#           POLY MODEL ACCURACY = 
np.mean(pred_test_poly==test_y) # Accuracy = 94.85


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
#           RBF MODEL ACCURACY = 
np.mean(pred_test_rbf==test_y) # Accuracy = 92.766


# kernel = sigmoid
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_X,train_y)
pred_test_sig = model_sig.predict(test_X)
#           sigmoid MODEL ACCURACY = 
np.mean(pred_test_sig==test_y) # Accuracy = 92.766


# kernel = precomputed
model_pre = SVC(kernel = "precomputed")
model_pre.fit(train_X,train_y)
pred_test_pre = model_pre.predict(test_X)
#           precomputed MODEL ACCURACY = 
np.mean(pred_test_pre==test_y) # Accuracy = 92.766