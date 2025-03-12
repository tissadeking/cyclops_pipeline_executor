from sklearn.cluster import DBSCAN
import numpy as np

def dbscan(X):
    # X = np.array([1, 4, 1, 0, 10, 2, 10, 4, 10, 0])
    X = np.array(X)
    epsi = 500
    # Create model
    dbscan = DBSCAN(eps=epsi, min_samples=2)

    dbscan.fit(X.reshape(-1, 1))

    return dbscan


