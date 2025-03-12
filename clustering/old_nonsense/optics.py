import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import DBSCAN

#data = pd.read_csv('Clustering_gmm.csv')
#X = [1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1.3, 4.1, 1.2, 0.5, 10.4, 2.2, 10.9, 4.5, 10.1, 0.3,
#              1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 1, 4, 1, 0, 10, 2, 10, 4, 10, 0, 20, 19, 15, 16, 9]
#print(len(X))
def optics_fun(X):
    X = np.array(X)
    X = X.reshape(-1, 1)
    # training gaussian mixture model
    # Building the OPTICS Clustering model
    optics_model = OPTICS(min_samples = 2, xi = 0.001, min_cluster_size = 0.05)
    # Training the model
    optics_model.fit(X)
    return optics_model
