# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:03:55 2020

@author: SHRAVANI PRAJIL
"""


import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

# Kmeans on airlines Data set 
airlines = pd.read_excel("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\airlines.xlsx")
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airlines.iloc[:,1:])


df_norm.head(10)  # Top 10 rows

###### screw plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
#elbow plot showing that it is bend at 5
#choose best cluster as 5
# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()

airlines = airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

airlines.iloc[:,11:13].groupby(airlines.clust).mean()

 #            Balance       Qual_miles   cc1_miles  cc2_miles
  #   clust                      
   #  0       83529.153046  290.453195   1.156018   1.032689   
    # 1       33097.301357   94.131783   1.070736   1.016473     
#     2      108317.387376  198.336634   3.915842   1.001238               
 #    3      118297.325243   73.467638   3.584142   1.001618   
  #   4       49921.633641   89.903226   1.122120   1.019585
     
   #          Bonus_miles  Bonus_trans  Flight_miles_12mo
 #   clust
#    0       8850.395245    10.476969        1030.112927
 #   1       3244.520349     6.173450         212.850775
  #  2      45609.657178    20.201733         713.728960
   # 3      31384.393204    17.233010         224.100324 
    #4       3467.074885     6.913594         243.834101   
    
       #    Days_since_enroll  Award?
  #  clust
#    0            4338.867756     1.0   
 #   1            1992.402132     0.0
  #  2            4863.439356     1.0
   # 3            4419.553398     0.0
    #4            5567.925115     0.0

airlines.to_csv("kmeansairlines.csv")
