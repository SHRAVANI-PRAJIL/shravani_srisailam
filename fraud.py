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

from sklearn.tree import  DecisionTreeClassifier
#help(DecisionTreeClassifier)
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(X_train,y_train)
preds = model.predict(X_test)
#preds is predicted values of test data
type(preds)
pd.Series(preds).value_counts()
np.mean(preds==y_test) # Accuracy of Test = 59.33%
np.mean(y_train == model.predict(X_train)) # Accuracy of train = 100%

#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, preds) 
cm

#Vizualising the Decision Trees few packages are required
###build a file to visualize 
#conda install -c conda-forge pydotplus

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import pydotplus
import matplotlib.pyplot as plt 
  
# reading png image file 
  
from pydotplus import graphviz
dot_data = StringIO()
export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = sel_features,class_names=['0','1'])
###visualize the .dot file. Need to install graphviz seperately at first that we have already done
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png("fraud.png") # to give a name to the tree
Image(graph.create_png()) # to vizualise the tree

###################################################################################################
#BAGGING technique
###################################################################################################
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
# accuracy of the model
train_acc = np.mean(bag_clf.predict(X_train)==y_train) # 81.11%
train_acc
test_pred = bag_clf.predict(X_test)
test_acc = np.mean(bag_clf.predict(X_test)==y_test) #75.33%
test_acc

###################################################################################################
# BOOSTING technique
###################################################################################################

import xgboost as xgb
### Preparing XGB classifier 
xgb1 = xgb.XGBClassifier(n_estimators=2000,learning_rate=0.3)
xgb1.fit(X_train,y_train)
train_pred = xgb1.predict(X_train)
import numpy as np
# accuracy of the model
train_acc = np.mean(train_pred==y_train) # 100%
train_acc
test_pred = xgb1.predict(X_test)
test_acc = np.mean(test_pred==y_test) #68.66%
test_acc
# Variable importance plot 
from xgboost import plot_importance
plot_importance(xgb1)
# f2 = city population is the most important feature