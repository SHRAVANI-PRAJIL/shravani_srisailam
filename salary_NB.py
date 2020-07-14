# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:21:04 2020

@author: SHRAVANI PRAJIL
"""


##################              Reading the Salary Data        ####################
salary_train = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\NB\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\NB\\SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
salary_train['Salary'] = salary_train['Salary'].apply(lambda x: 0 if x==' <=50K' else 1)
salary_test['Salary'] = salary_test['Salary'].apply(lambda x: 0 if x==' <=50K' else 1)

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

colnames = salary_train.columns
colnames
colnames = salary_train.columns
colnames
trainX = X_train = salary_train[['age','education']]
trainY = y_train = salary_train[colnames[13]]
testX = X_test  = salary_test[['age','education']]
testY = y_test  = salary_test[colnames[13]]
#trainX = salary_train[colnames[0:13]]
#trainY = salary_train[colnames[13]]
#testX  = salary_test[colnames[0:13]]
#testY  = salary_test[colnames[13]]
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 79.47%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+469+2920+780))  # 77.49%
### accuracy is good for Gaussian Naive Baye's 
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
y_train=sc_x.fit_transform(y_train)
y_test=sc_x.transform(y_test)
# Visualising the Training set results
#######################
# Visualising the results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, sgnb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'orange')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Education')
plt.legend()
plt.show()