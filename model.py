# model.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# For a differentiable approximation to DTW. 
# If you don't want the Soft-DTW approach, you can do a classic dynamic programming approach,
# but it won't be easily differentiable.
try:
    from tslearn.metrics import dtw as classic_dtw
    # or from dtaidistance import dtw as classic_dtw
    # or SoftDTW below
except ImportError:
    classic_dtw = None

class NearestNeighborDTW:
    """
    Very simple "model" that just memorizes the training set and
    uses DTW distance to find the nearest neighbor at inference time.
    WARNING: This can be extremely slow on large datasets.
    """
    def __init__(self, use_soft=False):
        # If you want to use a PyTorch-based Soft-DTW, set use_soft=True and
        # implement it in forward(). Otherwise, use a library function for classic DTW.
        self.use_soft = use_soft
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        """
        For each test sample, find the training sample with minimal DTW distance.
        Return predicted labels as an array of shape (len(X_test),).
        """
        y_pred = []
        for i, x in enumerate(X_test):
            best_dist = float("inf")
            best_label = None
            for j, x_train in enumerate(self.X_train):
                dist = self.dtw_distance(x, x_train)
                if dist < best_dist:
                    best_dist = dist
                    best_label = self.y_train[j]
            y_pred.append(best_label)
        return np.array(y_pred)

    def dtw_distance(self, x1, x2):
        if classic_dtw is None:
            raise RuntimeError("You need to install a DTW library, e.g. tslearn or dtaidistance.")
        return classic_dtw(x1, x2)


class SoftDTWLoss(nn.Module):
    """
    An example PyTorch-compatible Soft-DTW loss function.
    This code uses the differentiable approach so we can do K-means in PyTorch.
    """
    def __init__(self, gamma=0.01, use_cuda=False):
        super().__init__()
        self.gamma = gamma
        self.use_cuda = use_cuda

    def forward(self, x, y):
        """
        x: shape (batch_size, T, C)
        y: shape (batch_size, T, C)
        returns: sum of Soft-DTW distances
        """
        # naive example; you can find advanced implementations online
        # or use 'softdtw_cuda' from various repos.
        # Here we assume a placeholder that calls out to e.g. the "tslearn" SoftDTW
        # in a loop or something similar.
        loss_sum = 0.0
        for i in range(x.size(0)):
            xi = x[i].detach().cpu().numpy()
            yi = y[i].detach().cpu().numpy()
            dist = self.classic_dtw(xi, yi)
            loss_sum += dist
        return torch.tensor(loss_sum, requires_grad=True)

    def classic_dtw(self, x1, x2):
        if classic_dtw is None:
            raise RuntimeError("Install a DTW library, e.g. tslearn or dtaidistance.")
        return classic_dtw(x1, x2)


class KMeansSoftDTW(nn.Module):
    """
    Simple example of K-means with learnable prototypes and
    Soft-DTW used as distance.
    """
    def __init__(self, K=16, T=50, C=9, max_iter=10, lr=1e-3):
        """
        K: number of clusters
        T, C: time-series shape
        max_iter: number of gradient descent steps for learning prototypes
        """
        super().__init__()
        self.K = K
        self.T = T
        self.C = C
        self.max_iter = max_iter
        # Each prototype is shape (T, C)
        self.prototypes = nn.Parameter(torch.randn(K, T, C))
        self.sdtw_loss = SoftDTWLoss()
        self.lr = lr

    def forward(self, X):
        """
        Assign each time series in X to the closest prototype in DTW sense.
        X: shape (N, T, C)
        returns cluster_assignments: (N,).
        """
        with torch.no_grad():
            dist_matrix = []
            for k in range(self.K):
                # compute DTW distance of X to prototypes[k]
                dist_k = []
                for i in range(X.shape[0]):
                    dist_i = classic_dtw(X[i], self.prototypes[k].cpu().numpy())
                    dist_k.append(dist_i)
                dist_matrix.append(dist_k)
            dist_matrix = np.array(dist_matrix)  # shape (K, N)
            # find argmin over K
            cluster_assignments = np.argmin(dist_matrix, axis=0)
            return cluster_assignments

    def fit(self, X, max_epochs=1):
        """
        Gradient-descent approach to refine prototypes using Soft-DTW, reminiscent of K-means steps:
        1) Hard assignment by nearest prototype
        2) Minimization step w.r.t. prototypes
        Because DTW is expensive, we typically do fewer steps or use mini-batches.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        X_torch = torch.from_numpy(X).float().to(device)

        # Repeat for a small number of outer "K-means" iterations:
        for it in range(self.max_iter):
            # Step 1: Hard assignment (on CPU for simpler dtw here, or you can do a SoftDTW in GPU)
            cluster_assignments = self.forward(X)

            # Step 2: update prototypes by gradient descent
            # We'll build a small dataset for each cluster
            # Then do a step of gradient descent on each cluster's prototypes
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=device)

            for k in range(self.K):
                idx_k = np.where(cluster_assignments == k)[0]
                if len(idx_k) == 0:
                    continue

                X_k = X_torch[idx_k]  # (Nk, T, C)
                # We'll replicate the prototype for each sample
                prototype_k = self.prototypes[k].unsqueeze(0).expand(len(idx_k), self.T, self.C)
                # Soft-DTW loss
                loss_k = self.sdtw_loss(X_k, prototype_k)
                total_loss += loss_k

            total_loss.backward()
            optimizer.step()

            print(f"Iteration {it+1}/{self.max_iter}, total_loss={total_loss.item():.2f}")

