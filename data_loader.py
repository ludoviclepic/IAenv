import os
import numpy as np

class SatelliteTimeSeriesDataset:
    def __init__(self, root_dir, normalize=True):
        """
        Loads satellite image time series data from the given directory.
        Each subfolder in root_dir corresponds to a crop class and contains data files 
        for each sample (time series) with 9 spectral bands plus a flag column.
        """
        self.root_dir = root_dir
        self.normalize = normalize
        self.data = []   # list of time-series arrays
        self.labels = [] # list of class indices
        self.class_names = []
        
        # First pass: collect class names and data file paths
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            self.class_names.append(class_name)
            for file in sorted(os.listdir(class_folder)):
                file_path = os.path.join(class_folder, file)
                if file_path.endswith('.csv') or file_path.endswith('.txt'):
                    self._load_file(file_path, class_idx)
        
        self.data = np.array(self.data)   # shape: (N, T, C)
        self.labels = np.array(self.labels, dtype=int)
        
        # Normalize spectral bands (per band across dataset)
        if self.normalize:
            # Compute mean and std for each spectral band using training data only.
            # (We assume initially all loaded data is training; if not, call normalize() later with train indices.)
            data_vals = self.data.reshape(-1, self.data.shape[-1])  # combine all time steps
            self.band_means = data_vals.mean(axis=0)
            self.band_stds = data_vals.std(axis=0) + 1e-8
            # Apply normalization
            self.data = (self.data - self.band_means) / self.band_stds
    
    def _load_file(self, file_path, class_idx):
        """Loads a single time-series file, performs gap-filling interpolation, and stores it."""
        # Each file is assumed to have columns: possibly date, then 9 bands, then flag.
        # We'll ignore date if present and use index as time.
        raw = np.loadtxt(file_path, delimiter=',', skiprows=1)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        # Assume last column is flag (1 = valid observation, 0 = missing)
        values = raw[:, :-1]  # spectral values
        mask = raw[:, -1]     # binary flag
        T, C = values.shape
        
        # If time series length is less than the maximum (e.g., missing end days), we can pad to consistent length if needed.
        # Here we assume all series have same length T (e.g., 363 days for TimeSen2Crop).
        # If not, determine a global T_max across dataset and pad shorter series with mask=0.
        
        # Gaussian filtering for gap filling (Equation 11 in the paper)
        sigma = 7  # 7 days as per paper
        t_idx = np.arange(T)
        filled = np.zeros_like(values)
        # Gaussian weights for each possible time difference
        # We compute a Gaussian kernel across time indices for convolution
        kernel = np.exp(-0.5 * ((t_idx[:, None] - t_idx[None, :]) / sigma)**2)
        # Apply Gaussian weighted sum for each time t
        # Only consider actual data (mask=1) in the weighted sum.
        for t in range(T):
            w = kernel[t] * mask  # weight for each time point relative to t, masked by data availability
            if w.sum() > 0:
                filled[t] = (w[:, None] * values).sum(axis=0) / w.sum()
            else:
                # No data at all (unlikely if entire series missing)
                filled[t] = 0
        # Store the interpolated series and mask
        self.data.append(filled)
        self.labels.append(class_idx)
    
    def normalize_with_stats(self, means, stds):
        """Normalize the data using provided means and stds (useful for test set)."""
        self.data = (self.data - means) / stds

def train_test_split(dataset, test_fraction=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    Returns: train_dataset, test_dataset
    """
    np.random.seed(random_state)
    indices = np.arange(len(dataset.data))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_data = dataset.data[train_idx]
    train_labels = dataset.labels[train_idx]
    test_data = dataset.data[test_idx]
    test_labels = dataset.labels[test_idx]
    
    # Create new dataset objects for train and test
    train_set = SatelliteTimeSeriesDataset(dataset.root_dir, normalize=False)
    test_set = SatelliteTimeSeriesDataset(dataset.root_dir, normalize=False)
    train_set.data, train_set.labels, train_set.class_names = train_data, train_labels, dataset.class_names
    test_set.data, test_set.labels, test_set.class_names = test_data, test_labels, dataset.class_names
    if dataset.normalize:
        # Use training stats for normalization
        train_set.band_means, train_set.band_stds = dataset.band_means, dataset.band_stds
        test_set.band_means, test_set.band_stds = dataset.band_means, dataset.band_stds
        train_set.normalize = True
        test_set.normalize = True
        train_set.data = (train_set.data - train_set.band_means) / train_set.band_stds
        test_set.data = (test_set.data - test_set.band_means) / test_set.band_stds
    return train_set, test_set

# Example usage:
# dataset = SatelliteTimeSeriesDataset("path/to/dataset/root")
# train_set, test_set = train_test_split(dataset, test_fraction=0.1)
