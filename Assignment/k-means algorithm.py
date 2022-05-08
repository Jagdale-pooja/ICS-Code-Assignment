import numpy as np
import pandas as pd
import matplotlib

#create dataset
X=[[0.1,0.6],[0.15,0.71],[0.08,0.9],[0.16,0.85],[0.2,0.3],[0.25,0.5],[0.24,0.1],[0.3,0.2]]

#initial centroid points
centers=np.array([[0.1,0.6],[0.3,0.2]])
print("initial centroids: \n ",centers)

#import KMeans class
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,init=centers,n_init=1 )
model.fit(X)
print("labels:",model.labels_)

#Find P6 Location
print("P6 belongs to cluster",model.labels_[5])

#using labels find population around centroid
print("No. of population around cluster 2:",np.count_nonzero(model.labels_==1))


#find new centroids
print("New centroids:\n",model.cluster_centers_)


