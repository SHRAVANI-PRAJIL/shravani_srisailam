import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

# Kmeans on crime Data set 
crime = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\crime_data.csv")
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])
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

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()

crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.iloc[:,1:6].groupby(crime.clust).mean()
crime.to_csv("crimedataset.csv")


#selecting urban population and rape data from crime dataset

X = crime.iloc[:,[4,5]].values
X1=df_norm.iloc[:,[2,3]]
modelx = KMeans(n_clusters=5) 
modelx.fit(X1)
print(modelx.cluster_centers_)
centers = np.array(modelx.cluster_centers_)

Y = modelx.labels_ # getting the labels of clusters assigned to each row 
mdx = pd.Series(modelX.labels_)  # converting numpy array into pandas series object 
# Visualising the clusters
plt.scatter(X[Y==0, 0], X[Y==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[Y==1, 0], X[Y==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[Y==2, 0], X[Y==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[Y==3, 0], X[Y==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[md==4, 0], X[md==4, 1], s=100, c='magenta', label ='Cluster 5')
#Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.scatter(centers[:,0], centers[:,1], marker="x", color='yellow')

plt.title('Clusters of crime')
plt.xlabel('urban population')
plt.ylabel('rape cases')
plt.show()
