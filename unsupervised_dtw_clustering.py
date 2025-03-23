import numpy as np
from tqdm import tqdm

def dtw_distance(series1, series2):
    n, m = len(series1), len(series2)
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            diff = series1[i] - series2[j]
            cost[i, j] = np.sum(diff**2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            dtw[i, j] = cost[i-1, j-1] + min(dtw[i-1, j],
                                             dtw[i, j-1],
                                             dtw[i-1, j-1])
    return dtw[n, m]

class DTWClustering:
    def __init__(self, n_clusters, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
    
    def fit(self, X):
        N = X.shape[0]
        centroids_idx = np.random.choice(N, self.n_clusters, replace=False)
        self.centroids = [X[idx] for idx in centroids_idx]
        
        for iteration in tqdm(range(self.max_iter), desc="DTW Clustering Iterations"):
            clusters = {k: [] for k in range(self.n_clusters)}
            for i in tqdm(range(N), desc="Assigning clusters", leave=False):
                series = X[i]
                distances = [dtw_distance(series, centroid) for centroid in self.centroids]
                cluster_idx = int(np.argmin(distances))
                clusters[cluster_idx].append(i)
            new_centroids = []
            for k in range(self.n_clusters):
                if len(clusters[k]) == 0:
                    new_centroids.append(self.centroids[k])
                    continue
                cluster_series_idx = clusters[k]
                best_idx = None
                best_sum_dist = np.inf
                for idx in tqdm(cluster_series_idx, desc=f"Updating centroid {k}", leave=False):
                    d_sum = 0.0
                    for jdx in cluster_series_idx:
                        if idx == jdx:
                            continue
                        d_sum += dtw_distance(X[idx], X[jdx])
                    if d_sum < best_sum_dist:
                        best_sum_dist = d_sum
                        best_idx = idx
                new_centroids.append(X[best_idx] if best_idx is not None else self.centroids[k])
            converged = True
            for old, new in zip(self.centroids, new_centroids):
                if not np.array_equal(old, new):
                    converged = False
                    break
            self.centroids = new_centroids
            if converged:
                break
    
    def predict(self, X):
        labels = []
        N = X.shape[0]
        for i in tqdm(range(N), desc="Predicting clusters", leave=False):
            series = X[i]
            distances = [dtw_distance(series, centroid) for centroid in self.centroids]
            labels.append(int(np.argmin(distances)))
        return np.array(labels)
