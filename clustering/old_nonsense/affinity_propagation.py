import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from sklearn.cluster import AffinityPropagation
from statistics import mean

#data = pd.read_csv('Clustering_gmm.csv')
#X = [1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1.3, 4.1, 1.2, 0.5, 10.4, 2.2, 10.9, 4.5, 10.1, 0.3,
#              1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 20, 19, 15, 16, 34, 44, 33]
#epss = mean(X)/2
#print(len(X))

def affinity_fun(X):
    X = np.array(X)
    X = X.reshape(-1, 1)
    init = np.mean(pdist(X))
    # Compute Affinity Propagation
    #af = AffinityPropagation(preference=-50).fit(X)
    af = AffinityPropagation(preference=-1*init).fit(X)
    return af


'''cluster_centers_indices = af.cluster_centers_indices_
print('indices: ', cluster_centers_indices)
labels = af.labels_
print('labels: ', labels)
n_clusters_ = len(cluster_centers_indices)
print('n clusters: ', n_clusters_)

dbscan = DBSCAN(eps=epss, min_samples=2)
dbscan.fit(X.reshape(-1, 1))
print('dbscan.labels_: ', dbscan.labels_)'''
