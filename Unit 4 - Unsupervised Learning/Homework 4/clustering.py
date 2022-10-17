import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def plot_kmeans(x, z0):

    fig,ax =  plt.subplots()
    ax.grid(True)
    plt.scatter(x[:,0],x[:,1],c = 'b', alpha = 0.75)
    plt.scatter(z0[:, 0], z0[:, 1], c='r')
    plt.grid
    plt.show()

def kmeans(x,z0,norm_ord):
    # Step 1: Randomly select points --> already initialized in z0
    #    --

    # Step 2: Define the clusters
        ## Calculating Euclidean distance
        dist = np.empty(shape = (x.shape[0],1))
        for zk in z0:
            dif = x - zk
            d = (np.linalg.norm(dif,ord= norm_ord,axis=1)**2)
            dist = np.column_stack((dist,d))

        dist = dist[:,1:]
        ## Assign the cluster
        cluster = np.argmin(dist,axis=1)
        x = np.column_stack((x,cluster))

    # Step 3: Find the representatives

        ## Find the centroids
        z = np.empty(shape = (z0.shape[0],1),dtype=np.int64)
        for j in range(len(z0)):
            cj_idx = list(np.where(x[:,-1]  == j))
            centroid = (np.sum(x[cj_idx,:-1],axis = 1)/len(np.where(cluster == j))).T
            z = np.column_stack((z,centroid))

        return (cluster,z[:,1:])

def kmedoids(x,z0,norm_ord,max_iter):
    # Step 1: Randomly select points --> already initialized in z0
    #    --

    # Step 2: Define the clusters
    z = np.empty(shape=(z0.shape[0], 1), dtype=np.int64)
    ## iterate
    for k in range(max_iter):
        ## Calculating Euclidean distance
        dist1 = np.empty(shape=(x.shape[0], 1))
        for zk in z0:
            dif = x - zk
            d = (np.linalg.norm(dif,ord= norm_ord,axis=1)**2)
            dist1 = np.column_stack((dist1,d))

        dist1 = dist1[:,1:]
        ## Assign the cluster
        cluster = np.argmin(dist1,axis=1)
        x = np.column_stack((x,cluster))

    # Step 3: Find the representatives

        ## Find the centroids
        for j in range(len(z0.shape[0])):
            cj_idx = np.where(x[:, -1] == j)
            xj = x[cj_idx, :-1]
            for i,xi in enumerate(xj):
                x_rest = np.delete(i)
                dist2 = distance_matrix(xi,x_rest)






        z = z[:,1:]
        z0 = z

    return (cluster,z)

dict = {'A':1,'B':2}
print(list(dict.keys()))
