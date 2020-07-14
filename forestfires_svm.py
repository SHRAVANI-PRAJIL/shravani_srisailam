# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:38:47 2020

@author: SHRAVANI PRAJIL
"""
import pandas as pd 
import numpy as np 
import seaborn as sns
# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
##################              Reading the Salary Data        ####################
data = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\SVM\\forestfires.csv")
data.describe()
data.dtypes
data.head()
data = data.drop(['month','day'], axis=1)
data.info
data.dtypes

################           ASSIGNING INPUT OUTPUT VARIABLES      ###################
# Let's divide into train and test set
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
    data.drop(labels='size_category', axis=1),  # predictors
    data['size_category'],  # target
    test_size=0.2,
    random_state=0)

train_X.shape, test_X.shape

################                   standardizing data             ###################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

################                   dealing with imbalanced data             ###################
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(train_X,np.array( train_y).ravel())

################            Create SVM classification object                    ####################
#                  'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'                       #########
from sklearn.svm import SVC
# # Initialize SVM classifier for kernel = linear

help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
model_linear.fit(X_train_res,y_train_res)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 92.30%

###  after applying SMOTE accuracy increased from 92.30% to 95.19%
model_linear.fit(X_train_res,y_train_res)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 95.19%

# # Initialize SVM classifier for kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
#           POLY MODEL ACCURACY = 
np.mean(pred_test_poly==test_y) # Accuracy = 69.23%


## # Initialize SVM classifier for kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
#           RBF MODEL ACCURACY = 
np.mean(pred_test_rbf==test_y) # Accuracy = 76.92%


# # # Initialize SVM classifier for kernel = sigmoid
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_X,train_y)
pred_test_sig = model_sig.predict(test_X)
#           sigmoid MODEL ACCURACY = 
np.mean(pred_test_sig==test_y) # Accuracy = 81.73%


# # # Initialize SVM classifier for kernel = precomputed
model_pre = SVC(kernel = "precomputed")
model_pre.fit(train_X,train_y)
pred_test_pre = model_pre.predict(test_X)
#           precomputed MODEL ACCURACY = 
np.mean(pred_test_pre==test_y) # Accuracy = 92.766