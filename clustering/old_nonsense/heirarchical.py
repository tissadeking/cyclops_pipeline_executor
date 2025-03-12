import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from statistics import mean

#data = pd.read_csv('Clustering_gmm.csv')
#X = [1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1.3, 4.1, 1.2, 0.5, 10.4, 2.2, 10.9, 4.5, 10.1, 0.3,
#              1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 20, 19, 15, 16, 9]

def heirarchy_fun(X):
    thresh = mean(X)
    X = np.array(X)
    X = X.reshape(-1, 1)
    # training gaussian mixture model
    #data = list(zip(X, X))
    #print(thresh)
    hierarchical_cluster = AgglomerativeClustering(distance_threshold=thresh, n_clusters=None, affinity='euclidean', linkage='complete')
    hierarchical_cluster.fit(X)
    return hierarchical_cluster
