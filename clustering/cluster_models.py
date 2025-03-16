import numpy as np
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import pdist
from sklearn.cluster import AffinityPropagation
from statistics import mean
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

#X = [1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1.3, 4.1, 1.2, 0.5, 10.4, 2.2, 10.9, 4.5, 10.1, 0.3,
#              1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 20, 19, 15, 16, 34, 44, 33]

def affinity_fun(X):
    X = np.array(X)
    X = X.reshape(-1, 1)
    init = np.mean(pdist(X))
    # Compute Affinity Propagation
    #af = AffinityPropagation(preference=-50).fit(X)
    af = AffinityPropagation(preference=-1*init).fit(X)
    return af

def dbscan_fun(X):
    #X = np.array([1, 4, 1, 0, 10, 2, 10, 4, 10, 0])
    X = np.array(X)
    epsi = np.mean(X)/2 #500
    # Create model
    dbscan = DBSCAN(eps=epsi, min_samples=2)
    dbscan.fit(X.reshape(-1, 1))
    return dbscan

def gmm_fun(X):
    X = np.array(X)
    X = X.reshape(-1, 1)
    #clusters range
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    silhouette_avg = []
    cluster_labels = 0
    for num_clusters in range_n_clusters:
        # initialise kmeans
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        # silhouette score
        silhouette_avg.append(silhouette_score(X, cluster_labels))
    x = silhouette_avg.index(max(silhouette_avg))
    #print('x: ', x)
    n_comp = range_n_clusters[x]
    n_components = n_comp
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X)
    return gmm

def heirarchy_fun(X):
    thresh = mean(X)
    X = np.array(X)
    X = X.reshape(-1, 1)
    hierarchical_cluster = AgglomerativeClustering(distance_threshold=thresh, n_clusters=None, affinity='euclidean', linkage='complete')
    hierarchical_cluster.fit(X)
    return hierarchical_cluster

def km_fun(X):
    X = np.array(X)
    X = X.reshape(-1, 1)
    #clusters range
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    silhouette_avg = []
    cluster_labels = 0
    for num_clusters in range_n_clusters:
        # initialise kmeans
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        # silhouette score
        silhouette_avg.append(silhouette_score(X, cluster_labels))

    x = silhouette_avg.index(max(silhouette_avg))
    #print('x: ', x)
    n_comp = range_n_clusters[x]
    kmeans = KMeans(n_clusters=n_comp)
    kmeans.fit(X)
    return kmeans

def optics_fun(X):
    X = np.array(X)
    X = X.reshape(-1, 1)
    # Building the OPTICS Clustering model
    optics_model = OPTICS(min_samples = 2, xi = 0.001, min_cluster_size = 0.05)
    # Training the model
    optics_model.fit(X)
    return optics_model


