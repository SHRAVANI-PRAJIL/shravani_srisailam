# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:26:43 2020

@author: SHRAVANI PRAJIL
"""


import pandas as pd
import matplotlib.pylab as plt 
crime = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\clustering\\crime_data.csv")
# Normalization function 
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
from	sklearn.cluster	import	AgglomerativeClustering 
type(df_norm)

########  building model by euclidean distance and methos as complete ##########################
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms for method = complete ")  
dend = shc.dendrogram(shc.linkage(df_norm, method='complete'))
plt.axhline(y=3.5, color='r', linestyle='--')
# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(df_norm) 
h_complete.labels_

########  building model by euclidean distance and methos as single ##########################
z1 = linkage(df_norm, method="single",metric="euclidean")
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms for method = single ")  
dend = shc.dendrogram(shc.linkage(df_norm, method='single'))
plt.axhline(y=1.35, color='r', linestyle='--')
# Now applying AgglomerativeClustering choosing 2 as clusters from the above dendrogram
h_complete1	=	AgglomerativeClustering(n_clusters=4,	linkage='single',affinity = "euclidean").fit(df_norm) 
h_complete1.labels_

########  building model by euclidean distance and method as ward ##########################
z2 = linkage(df_norm, method="ward",metric="euclidean")
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms for method = ward ")  
dend = shc.dendrogram(shc.linkage(df_norm, method='ward'))
plt.axhline(y=8, color='r', linestyle='--')
# Now applying AgglomerativeClustering choosing 2 as clusters from the above dendrogram
h_complete2	=	AgglomerativeClustering(n_clusters=2,	linkage='ward',affinity = "euclidean").fit(df_norm) 
h_complete2.labels_

########  building model by euclidean distance and method as centroid ##########################
z3 = linkage(df_norm, method="centroid",metric="euclidean")
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms for method = centroid ")  
dend = shc.dendrogram(shc.linkage(df_norm, method='centroid'))
plt.axhline(y=1.6, color='r', linestyle='--')
# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
h_complete3	=	AgglomerativeClustering(n_clusters=4,	linkage='centroid',affinity = "euclidean").fit(df_norm) 
h_complete3.labels_

########  building model by euclidean distance and method as average ##########################
z4 = linkage(df_norm, method="average",metric="euclidean")
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms for method = average ")  
dend = shc.dendrogram(shc.linkage(df_norm, method='average'))
plt.axhline(y=2, color='r', linestyle='--')
# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
h_complete4	=	AgglomerativeClustering(n_clusters=4,	linkage='average',affinity = "euclidean").fit(df_norm) 
h_complete4.labels_
###############################################################################################
################   distance method = Manhattan ###############################################
########  building model by Manhattan distance and method as average ##########################
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
z5 = pairwise_distances(df_norm, metric='manhattan')
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms for method = average ")  
dend = shc.dendrogram(shc.linkage(df_norm, method='average'))
plt.axhline(y=2, color='r', linestyle='--')
clustering = SpectralClustering(n_clusters=4, affinity='precomputed', assign_labels="discretize",random_state=0)
clustering.fit(z5)
clustering.labels_
clustering 

########  building model by Manhattan distance and method as complete ##########################
z5 = pairwise_distances(df_norm, metric='manhattan')
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms for method = complete ")  
dend = shc.dendrogram(shc.linkage(df_norm, method='complete'))
plt.axhline(y=3.5, color='r', linestyle='--')
clustering = SpectralClustering(n_clusters=4, affinity='precomputed', assign_labels="discretize",random_state=0)
clustering.fit(z5)
clustering.labels_
clustering 
###############################################################################################
################   distance method = cosine ###############################################
########  building model by cosine distance and method as average ##########################
z6 = pairwise_distances(df_norm, metric='cosine')
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms for method = average ")  
dend = shc.dendrogram(shc.linkage(df_norm, method='average'))
plt.axhline(y=2.8, color='r', linestyle='--')
clustering = SpectralClustering(n_clusters=4, affinity='precomputed', assign_labels="discretize",random_state=0)
clustering.fit(z5)
clustering.labels_
clustering 



labels = h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
labels = h_complete.labels_
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.head()

# getting aggregate mean of each cluster
crime.groupby(crime.clust).mean()

# creating a csv file 
crime.to_csv("crimedata.csv") #,encoding="utf-8")

