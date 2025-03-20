import numpy as np

def dtw_distance(series1, series2):
    """Compute Dynamic Time Warping distance between two time series (as numpy arrays of shape (T, C))."""
    n, m = len(series1), len(series2)
    # If multivariate, we define distance at each time as sum of squared differences across bands
    # Compute cost matrix
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # Euclidean distance between vectors at i and j
            diff = series1[i] - series2[j]
            cost[i, j] = np.sum(diff**2)
    # DTW dynamic programming
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            dtw[i, j] = cost[i-1, j-1] + min(dtw[i-1, j],    # deletion
                                             dtw[i, j-1],    # insertion
                                             dtw[i-1, j-1])  # match
    return dtw[n, m]

class DTWClustering:
    def __init__(self, n_clusters, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None  # will store indices of centroid series or the series themselves
    
    def fit(self, X):
        """
        X: numpy array of shape (N, T, C) containing N time series.
        We'll perform clustering and store cluster centroids (as actual time series).
        """
        N = X.shape[0]
        # Randomly choose initial centroids (by index)
        centroids_idx = np.random.choice(N, self.n_clusters, replace=False)
        self.centroids = [X[idx] for idx in centroids_idx]
        
        for iteration in range(self.max_iter):
            # Assignment step
            clusters = {k: [] for k in range(self.n_clusters)}
            for i in range(N):
                series = X[i]
                # Find nearest centroid by DTW distance
                distances = [dtw_distance(series, centroid) for centroid in self.centroids]
                cluster_idx = int(np.argmin(distances))
                clusters[cluster_idx].append(i)
            # Update step
            new_centroids = []
            for k in range(self.n_clusters):
                if len(clusters[k]) == 0:
                    # If a cluster lost all points, reassign a random series as centroid
                    new_centroids.append(self.centroids[k])
                    continue
                # Find medoid: the series in cluster k with minimum average distance to others
                cluster_series_idx = clusters[k]
                best_idx = None
                best_sum_dist = np.inf
                for idx in cluster_series_idx:
                    # Compute total distance from series idx to all others in cluster
                    d_sum = 0.0
                    for jdx in cluster_series_idx:
                        if idx == jdx:
                            continue
                        d_sum += dtw_distance(X[idx], X[jdx])
                    if d_sum < best_sum_dist:
                        best_sum_dist = d_sum
                        best_idx = idx
                new_centroids.append(X[best_idx] if best_idx is not None else self.centroids[k])
            # Check convergence (if centroids didn't change)
            converged = True
            for old, new in zip(self.centroids, new_centroids):
                if not np.array_equal(old, new):
                    converged = False
                    break
            self.centroids = new_centroids
            if converged:
                break
    
    def predict(self, X):
        """Assign each time series in X to the nearest centroid (return cluster indices)."""
        labels = []
        for series in X:
            distances = [dtw_distance(series, centroid) for centroid in self.centroids]
            labels.append(int(np.argmin(distances)))
        return np.array(labels)
