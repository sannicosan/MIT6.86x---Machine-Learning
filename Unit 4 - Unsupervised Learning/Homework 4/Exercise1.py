# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import clustering as clust
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import  KMeans
from mlinsights.mlmodel import KMeansL1L2

x = np.array([[0,-6],[4,4],[0,0],[-5,2]])
z0 = np.array([[-5,2],[0,-6]])

##region Clustering 1
# clust.plot_kmeans(x, z0)
print('------------------- Clustering 1 --------------------')
kmedoids = KMedoids(n_clusters= 2,metric='l1',init = 'random').fit(x)
print('Cluster labels: ',kmedoids.labels_)
print('\n Centroids: \n', kmedoids.cluster_centers_)
print('------------------------------------------------------\n')
##endregion


##region Clustering 2
# clust.plot_kmeans(x, z0)
print('------------------- Clustering 2 --------------------')
kmedoids = KMedoids(n_clusters= 2,metric='l2',init='random').fit(x)
print('Cluster labels: ',kmedoids.labels_)
print('\n Centroids: \n', kmedoids.cluster_centers_)
print('------------------------------------------------------\n')
##endregion

##region Clustering 3
# clust.plot_kmeans(x, z0)
print('------------------- Clustering 3 --------------------')
kmeans = KMeansL1L2(n_clusters= 2,norm = 'L1',max_iter = 300).fit(x)
print('Cluster labels: ',kmeans.labels_)
print('\n Centroids: \n', kmeans.cluster_centers_)
print('------------------------------------------------------\n')
##endregion

