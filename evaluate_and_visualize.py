import numpy as np
from sklearn.metrics import confusion_matrix
import torch

def evaluate_supervised(model, test_set):
    model.eval()
    X = torch.tensor(test_set.data, dtype=torch.float32)
    y_true = test_set.labels
    with torch.no_grad():
        recon_all, _, _ = model(X)  # (N, K, T, C)
        # Compute MSE to each prototype for each sample
        mse = ((recon_all - X.unsqueeze(1))**2).mean(dim=(2,3)).cpu().numpy()  # (N, K)
    y_pred = np.argmin(mse, axis=1)
    # Overall Accuracy
    OA = (y_pred == y_true).mean()
    # Mean Accuracy (average per-class accuracy)
    conf_mat = confusion_matrix(y_true, y_pred, labels=range(model.num_classes))
    per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    MA = np.nanmean(per_class_acc)  # mean of per-class accuracy
    return OA, MA, y_pred

def evaluate_clustering(cluster_labels, true_labels):
    # Map each cluster to the most frequent true class
    cluster_to_class = {}
    for c in np.unique(cluster_labels):
        mask = (cluster_labels == c)
        if np.any(mask):
            # majority vote
            votes = true_labels[mask]
            if len(votes) == 0:
                cluster_to_class[c] = -1
            else:
                vals, counts = np.unique(votes, return_counts=True)
                cluster_to_class[c] = vals[np.argmax(counts)]
    # Predict mapped classes
    mapped_pred = np.array([cluster_to_class[c] for c in cluster_labels])
    # Overall Accuracy
    OA = (mapped_pred == true_labels).mean()
    # Mean Accuracy
    conf_mat = confusion_matrix(true_labels, mapped_pred, labels=np.unique(true_labels))
    per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    MA = np.nanmean(per_class_acc)
    return OA, MA, mapped_pred

# Visualization of prototypes and alignments
import matplotlib.pyplot as plt

def plot_prototype_reconstruction(model, sample, true_label):
    """Plot an input sample vs the corresponding prototype before and after alignment."""
    model.eval()
    x = torch.tensor(sample[None, ...], dtype=torch.float32)  # shape (1, T, C)
    with torch.no_grad():
        # Get reconstruction for true class prototype
        recon_all, _, _ = model(x)
    recon = recon_all[0, true_label].cpu().numpy()  # prototype reconstruction for the true class
    prototype = model.prototypes[true_label].cpu().numpy()
    time = np.arange(sample.shape[0])
    # Plot one spectral band (e.g., first band) for clarity
    plt.figure(figsize=(6,4))
    plt.plot(time, sample[:, 0], label='Input (Band 1)', color='blue')
    plt.plot(time, prototype[:, 0], label='Prototype (Band 1)', color='orange', linestyle='--')
    plt.plot(time, recon[:, 0], label='Aligned Prototype (Band 1)', color='green', linestyle='-.')
    plt.title(f"Prototype Reconstruction for Class {model.class_names[true_label]}")
    plt.xlabel("Time")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.show()

def plot_dtw_alignment(series1, series2):
    """Plot two time series with lines indicating DTW alignment."""
    # For visualization, offset series2 vertically
    off = 1.0
    s1 = series1[:, 0]  # first band
    s2 = series2[:, 0]  # first band
    plt.figure(figsize=(6,4))
    plt.plot(s1, label='Series 1', color='blue')
    plt.plot(s2 - off, label='Series 2 (offset)', color='red', linestyle='--')
    # Compute DTW path for visualization
    n, m = len(s1), len(s2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0,0] = 0
    # Simple DTW on first band for path
    for i in range(1, n+1):
        for j in range(1, m+1):
            dist = (s1[i-1] - s2[j-1])**2
            dtw[i,j] = dist + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    # Backtrack path
    i, j = n, m
    path = []
    while i>0 and j>0:
        path.append((i-1, j-1))
        step = np.argmin([dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1]])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    # Plot alignment lines
    for (i, j) in path:
        plt.plot([i, j], [s1[i], s2[j] - off], color='gray', linewidth=0.8)
    plt.title("DTW Alignment between two series")
    plt.legend()
    plt.show()
