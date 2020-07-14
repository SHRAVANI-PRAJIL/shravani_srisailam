# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:06:22 2020

@author: SHRAVANI PRAJIL
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:57:17 2020

@author: SHRAVANI PRAJIL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\decision tree\\Fraud_check.csv")
data.columns
data.head()
data['Taxable.Income'].unique()
data['Taxable.Income'] = np.where(data['Taxable.Income']<=30000, '0', '1') 
data['Taxable.Income'].unique()
data['Taxable.Income'].value_counts()
data.dtypes

from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
data["Undergrad"] = lb_make.fit_transform(data["Undergrad"])
data["Undergrad"].unique()
data["Marital.Status"] = lb_make.fit_transform(data["Marital.Status"])
data["Marital.Status"]
data["Urban"] = lb_make.fit_transform(data["Urban"])
data["Urban"]
data.dtypes
#feature Scaling  
predictors = data[['Undergrad', 'Marital.Status', 'City.Population', 'Work.Experience', 'Urban']]
sel_features = predictors.columns
target = data['Taxable.Income']
predictors.dtypes
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
predictors= st_x.fit_transform(predictors)    
target= st_x.transform(target)
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.25, random_state = 0)
#train,test = train_test_split(data,test_size = 0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
rf_y = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=100,criterion="entropy")
#rf_y.fit(trainX,trainY) # Error Can not convert a string into float means we have to use LabelEncoder()
# Considering only the string data type columns and 

from sklearn import preprocessing
for i in sel_features:
    number = preprocessing.LabelEncoder()
    X_train[i] = number.fit_transform(X_train[i])

rf_y.fit(X_train,y_train)
#rf_y_test is predicted values of test data
rf_y_test = rf_y.predict(X_test)
rf_y_train = rf_y.predict(X_train)
# Accuracy
type(rf_y_test)
pd.Series(rf_y_test).value_counts()
np.mean(rf_y_test==y_test) # Accuracy of Test = 74%
np.mean(y_train == rf_y_train) # Accuracy of train = 100%
#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, rf_y_test) 
cm

#Vizualising the Decision Trees few packages are required
###build a file to visualize 
#conda install -c conda-forge pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import pydotplus
import matplotlib.pyplot as plt 
import matplotlib.image as img 
  
# reading png image file 
  
from pydotplus import graphviz
dot_data = StringIO()
export_graphviz(rf_y, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = sel_features,class_names=['0','1'])
###visualize the .dot file. Need to install graphviz seperately at first that we have already done
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png("fraud.png") # to give a name to the tree
Image(graph.create_png()) # to vizualise the tree
