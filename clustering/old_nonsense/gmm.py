import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import KMeans

# data = pd.read_csv('Clustering_gmm.csv')
# X = np.array([1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1.3, 4.1, 1.2, 0.5, 10.4, 2.2, 10.9, 4.5, 10.1, 0.3,
#              1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 20, 19, 15, 16])
# X = [0, 0, 0, 0, 0, 0]

def gmm_fun(X):
    X = np.array(X)
    X = X.reshape(-1, 1)
    # training gaussian mixture model
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
    #print(silhouette_avg)
    # for i in range(len(range_n_clusters)):
    # range_n_clusters[
    #print('clus labels: ', len(np.unique(cluster_labels)))
    x = silhouette_avg.index(max(silhouette_avg))
    #print('x: ', x)
    n_comp = range_n_clusters[x]
    n_components = n_comp
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X)
    return gmm

'''#predictions from gmm
labels = gmm.predict(X)
print(gmm.means_)
#print('labels: ', labels)
frame = pd.DataFrame(X)
frame['cluster'] = labels
#frame.columns = ['Weight', 'Height', 'cluster']
frame.columns = ['weight', 'cluster']
print('frame: ', frame)

cluster_arr = []
for i in range(len(gmm.means_)):
    cluster_arr.append(gmm.means_[i][0])
cluster_arr.sort()
print('cluster arr: ', cluster_arr)
cluster_diff_arr = []
if n_components > 2:
    for i in range(len(cluster_arr) - 1):
        cluster_diff_arr.append(abs(cluster_arr[i] - cluster_arr[i + 1]))
    diff = abs(max(cluster_diff_arr) - min(cluster_diff_arr))
# diff = statistics.mean(cluster_diff_arr)
# diff = abs(max(cluster_diff_arr))
else:
    diff = abs(max(cluster_arr) - min(cluster_arr))
    # diff = statistics.mean(cluster_arr)
    # diff = abs(max(cluster_arr) - min(cluster_arr))
    #return diff
print('cluster diff arr: ', cluster_diff_arr)
print('diff: ', diff)'''

