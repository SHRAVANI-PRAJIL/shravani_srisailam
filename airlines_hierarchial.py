# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:42:21 2020

@author: SHRAVANI PRAJIL
"""


import pandas as pd
import matplotlib.pylab as plt 
airlines = pd.read_excel("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\airlines.xlsx")


# Normalization function 
#def norm_func(i):
 #   x = (i-i.min())	/	(i.max()	-	i.min())
  #  return (x)

# alternative normalization function 

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airlines.iloc[:,1:])

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")


plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=5,	linkage='complete',affinity = "euclidean").fit(df_norm) 

h_complete.labels_

cluster_labels=pd.Series(h_complete.labels_)

airlines['clust']=cluster_labels # creating a  new column and assigning it to new column 
airlines.head()
airlines = airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
airlines.head()

# getting aggregate mean of each cluster
mean = airlines.groupby(airlines.clust).mean()

#clust	Balance	            Qual_miles	        cc1_miles	         cc2_miles
#0	  117123.66470588236	255.75294117647059	2.2529411764705882	 1.3411764705882352
#1	  65902.07165520889   	137.3707033315706	2.0335801163405605	 1.0
#2	  138061.4	            78.8	            3.466666666666667	 1.0
#3	  131999.5	            347.0	            2.5	                 1.0

#clust	cc3_miles	          Bonus_miles	       Bonus_trans	        Flight_miles_12mo
#0   	1.0                   37437.17058823529	   26.729411764705883	4066.6235294117646
#1	    1.000793231094659	  15571.369910100477   10.724484399788471	270.5854045478583
#2   	4.066666666666666	  93927.86666666667	   28.066666666666666	506.6666666666667
#3   	1.0	                  65634.25	           69.25	            19960.0
#4   	1.0	                  58412.32142857143	   21.214285714285715	1344.392857142857

#clust	Flight_trans_12	       Days_since_enroll	    Award?
#0	    11.882352941176471	   4701.6882352941175      	0.7058823529411765
#1   	0.8183500793231094	   4072.2945531464834	    0.3503437334743522
#2   	1.6	                   4613.866666666667        0.5333333333333333
#3   	49.25	               2200.25	                1.0
#4   	5.607142857142857	   6835.892857142857	    0.8571428571428571

# creating a csv file 
airlines.to_csv("eastwestairlines.csv") #,encoding="utf-8")
