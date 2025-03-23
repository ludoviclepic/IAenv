import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from data_loader import SatelliteTimeSeriesDataset, train_test_split
from supervised_model import DeformableNCC, training_loop
from unsupervised_dtw_clustering import DTWClustering, dtw_distance
from evaluate_and_visualize import evaluate_supervised, evaluate_clustering, plot_prototype_reconstruction, plot_dtw_alignment

# ===== Data Loading =====
from data_loader import SatelliteTimeSeriesDataset, train_test_split

data_root = "/Users/ludoviclepic/Downloads/TimeSen2Crop-2"
dataset = SatelliteTimeSeriesDataset(data_root, normalize=True)
train_set, test_set = train_test_split(dataset, test_fraction=0.2)

# ===== Prepare DataLoader for Supervised Model =====
train_tensor = torch.tensor(train_set.data, dtype=torch.float32)
train_labels = torch.tensor(train_set.labels, dtype=torch.long)
train_dataset = TensorDataset(train_tensor, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ===== Supervised Model =====
from supervised_model import DeformableNCC, training_loop
num_classes = len(train_set.class_names)
series_length = train_set.data.shape[1]
num_bands = train_set.data.shape[2]
model = DeformableNCC(num_classes, series_length, num_bands)
model.class_names = train_set.class_names

training_loop(model, train_loader, num_epochs=20, lr=1e-3, mu=1.0, nu=0.01)

# ===== Evaluate Supervised Model =====
from evaluate_and_visualize import evaluate_supervised, plot_prototype_reconstruction
OA_supervised, MA_supervised, y_pred = evaluate_supervised(model, test_set)
print("Supervised Model Results:")
print(f"Overall Accuracy: {OA_supervised*100:.2f}%")
print(f"Mean Accuracy: {MA_supervised*100:.2f}%")
sample_idx = 0
sample = test_set.data[sample_idx]
true_label = test_set.labels[sample_idx]
plot_prototype_reconstruction(model, sample, true_label)

# ===== Unsupervised Model: DTW Clustering =====
from unsupervised_dtw_clustering import DTWClustering
from evaluate_and_visualize import evaluate_clustering, plot_dtw_alignment
import random

X_unsup = test_set.data
unsup_model = DTWClustering(n_clusters=num_classes, max_iter=5)
unsup_model.fit(X_unsup)
cluster_labels = unsup_model.predict(X_unsup)

true_labels = np.array(test_set.labels)
OA_unsup, MA_unsup, mapped_pred = evaluate_clustering(cluster_labels, true_labels)
print("\nUnsupervised DTW Clustering Results:")
print(f"Overall Accuracy: {OA_unsup*100:.2f}%")
print(f"Mean Accuracy: {MA_unsup*100:.2f}%")

idx1, idx2 = random.sample(range(X_unsup.shape[0]), 2)
series1 = X_unsup[idx1]
series2 = X_unsup[idx2]
plot_dtw_alignment(series1, series2)
