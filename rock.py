from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np

class ROCK:
    def __init__(self, tmax):
        self.tmax = tmax

    def fit(self, X):
        self.X = X
        eps = self.calculate_eps()
        original_size = X.shape[0]
        for t in range(0, self.tmax):
            k = int(np.rint(((0.5 * original_size - 3) / self.tmax) * t + 3))
            neigh = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
            neigh.fit(self.X)
            knn = neigh.kneighbors(self.X, return_distance=False)[:, 1:]
            knn_points = self.X[knn]
            X_new = np.mean(knn_points, axis=1)
            check = np.linalg.norm(X_new - self.X, axis=1)
            if np.where(check < eps)[0].shape[0] == 0:
                break
            self.X = X_new
        
        cluster = DBSCAN(eps=eps, min_samples=1)
        cluster.fit(self.X)
        self.labels = cluster.labels_
        return self

    def calculate_eps(self):
        neigh = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1)
        neigh.fit(self.X)
        return np.mean(neigh.kneighbors(self.X, return_distance=True)[0][:, 1])

    @property
    def labels_(self):
        return self.labels