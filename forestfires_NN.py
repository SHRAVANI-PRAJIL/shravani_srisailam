# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:38:47 2020

@author: SHRAVANI PRAJIL
"""
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

##################              Reading the Salary Data        ####################
data = pd.read_csv("C:\\Users\\HP\\Desktop\\EXCELR\\ASSIGNMENTS\\NN\\forestfires.csv")
data.describe()
data.dtypes
data.head()
data = data.drop(['month','day'], axis=1)
data.info
data.dtypes
data['size_category'].unique() 


data.loc[data.size_category=="large","size_category"] = 1
data.loc[data.size_category=="small","size_category"] = 0
data.size_category.value_counts().plot(kind="bar")
 
data['size_category'].unique() 

################           ASSIGNING INPUT OUTPUT VARIABLES      ###################
# Let's divide into train and test set
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2,random_state=42)
trainX = train.drop(["size_category"],axis=1)
trainY = train["size_category"]
testX = test.drop(["size_category"],axis=1)
testY = test["size_category"]
trainX, X_val, trainY, y_val = train_test_split(trainX, trainY, test_size=0.2, random_state=1)

trainX.shape, testX.shape
trainY.shape, testY.shape
X_val.shape, y_val.shape

trainX = np.asarray(trainX).astype(np.float32)
trainY = np.asarray(trainY).astype(np.float32)
X_val = np.asarray(X_val).astype(np.float32)
y_val = np.asarray(y_val).astype(np.float32)
   
model = Sequential([
        Dense(50, input_shape=(28,), activation='sigmoid'),
        Dense(30, input_shape=(50,), activation='sigmoid'),
        Dense(1, input_shape=(30,)),
    ])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(28,)),
#    keras.layers.Dense(50, activation=tf.nn.relu),
# 	keras.layers.Dense(30, activation=tf.nn.relu),
#    keras.layers.Dense(1, activation=tf.nn.sigmoid),
#      ])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])


history = model.fit(trainX, trainY, epochs=34, batch_size=1, validation_data=(X_val, y_val))

history_dict = history.history
history_dict.keys()
#### dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,35)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,35)
plt.plot(epochs, loss_train, 'r', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#import os
#os.environ['PATH'] = r'c:\\users\\hp\\anaconda64bit\\lib\\site-packages\\graphviz'
#os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+'c:\\users\\hp\\anaconda64bit\\lib\\site-packages\\graphviz'

#os.chdir(r'c:\\users\\hp\\anaconda64bit\\lib\\site-packages')
from ann_visualizer.visualize import ann_viz;
ann_viz(model, title="My first neural network")

