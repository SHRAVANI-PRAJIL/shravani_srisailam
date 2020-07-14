# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:05:31 2020

@author: SHRAVANI PRAJIL
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:04:46 2020

@author: SHRAVANI PRAJIL
"""

# In[1]:


#First of all, the packages that are needed for this data has to be imported
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier # to import DecisionTree Classifier
from sklearn.model_selection import train_test_split # For training & testing split function
from sklearn import metrics #for calculating accuracy


# In[2]:
#Load the dataset to start working on it
data = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\decision tree\\Company_Data.csv")
data.columns
data.head()
#col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'outcome']
data['Sales'] = np.where((data['Sales'].values) <=8, '0', '1') 
data['Sales'].unique()
data['Sales'].value_counts()

data.dtypes
#defining the columns
colnames = list(data.columns)
colnames
# In[3]:
#converting catagorical data
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
data["ShelveLoc"] = lb_make.fit_transform(data["ShelveLoc"])
data["ShelveLoc"].unique()
data["Urban"] = lb_make.fit_transform(data["Urban"])
data["Urban"]
data["US"] = lb_make.fit_transform(data["US"])
data["US"]
data.dtypes

#This function is used to get some description about the data
data.describe()


# In[4]:


#info() gives the information of the dataset like no. of columns,datatypes, range, class etc.
data.info()


# In[5]:


#Preprocessing, Feature Selection is carried out next to further proceed for building the model
data.isnull().sum() #checks if there is any NA values are present


# In[7]:
#defining train and test datasets

train=data.iloc[:,1:11] #Input Variables
test =data.iloc[:,0] #output Variable

feature_cols = colnames[1:11]

# In[8]:


#splitting the dataset into train & test
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size = 0.2,random_state = 51) 


# In[9]:
from sklearn.ensemble import RandomForestClassifier
#time to create the model via DecisionTree Classifier
rf_y = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=100,criterion="entropy")
rf_y.fit(X_train,y_train) 


# In[10]:

#Evaluating how the test data responses
#rf_y_test is predicted values of test data
rf_y_test = rf_y.predict(X_test)
rf_y_train = rf_y.predict(X_train)


# In[11]:
#preds is predicted values of test data
# Accuracy
type(rf_y_test)
pd.Series(rf_y_test).value_counts()
np.mean(rf_y_test==y_test) # Accuracy of Test = 85%
np.mean(y_train == rf_y_train) # Accuracy of train = 100%
#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, rf_y_test) 
cm

# In[12]:

#Vizualising the Decision Trees few packages are required
###build a file to visualize 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import pydotplus

dot_data = StringIO()
export_graphviz(rf_y, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
###visualize the .dot file. Need to install graphviz seperately at first that we have already done
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[13]:

graph.write_png("company sales.png") # to give a name to the tree
Image(graph.create_png()) # to vizualise the tree



