import pandas as pd 
import numpy as np
wine = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\PCA\\wine.csv")
wine.describe()
wine.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
wine.data = uni.iloc[:,1:]
wine.data.head(4)
#droping type cloumn
x = wine.drop('Type',1)
y=wine.iloc[:,0]
x
y
# Normalizing the numerical data 
wine_normal = scale(x)
pca = PCA()
pca_values = pca.fit_transform(wine_normal)
pca_values.shape

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 
x = np.array(pca_values[:,0])
y = np.array(pca_values[:,1])
z = np.array(pca_values[:,2])
plt.plot(x,y,"bo")

################### performing clustering for the first three PCA components  ##########################
new_df = pd.DataFrame(pca_values[:,0:3])
new_df
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
########from the elbow curve the optimum number of clusters is choosen as 3##########################
kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(kmeans.labels_)  # converting numpy array into pandas series object 
wine['clust']=md # creating a  new column and assigning it to new column 
wine.head()
wine = wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

################## Correlation between clustering grouping and initial grouping##################
r = np.corrcoef(wine.Type, wine.clust)
r
##### a strong positive correlation 0.96861116 ##################################################
################################################################################################

wine.iloc[:,2:15].groupby(wine.clust).mean()
wine.iloc[:,2:15].groupby(wine.Type).mean()

############ the clustering based on kmeans by considering the first 3PCA components########
############ and the intial grouping with 3 clusters########################################
#####       Alcohol     Malic       Ash  ...       Hue  Dilution      Proline
#####Type                                 ...                                 
#####1     13.744746  2.010678  2.455593  ...  1.062034  3.157797  1115.711864
#####2     12.278732  1.932676  2.244789  ...  1.056282  2.785352   519.507042
#####3     13.153750  3.333750  2.437083  ...  0.682708  1.683542   629.895833

#####[3 rows x 13 columns]
##### 
#####         Alcohol     Malic       Ash  ...       Hue  Dilution      Proline
#####clust                                 ...                                 
#####0      13.656032  1.983175  2.460476  ...  1.065079  3.157143  1093.238095
#####1      12.249062  1.910312  2.233281  ...  1.063063  2.803906   507.828125
#####2      13.134118  3.307255  2.417647  ...  0.691961  1.696667   619.058824

#####[3 rows x 13 columns]