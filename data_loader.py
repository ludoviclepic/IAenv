import os
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bar

class SatelliteTimeSeriesDataset:
    def __init__(self, root_dir, normalize=True):
        """
        Loads satellite image time series data from a structured directory.
        
        Expected structure:
        
          main_folder/
              region1/
                  1/      <- crop class folder (e.g., "1")
                      0.csv
                      1.csv
                      ...
                  2/
                  ...
                  15/
                  dates.csv   <-- important: contains acquisition dates
              region2/
                  1/
                      ...
                  dates.csv
              ...
        
        This loader:
          1. Reads each region’s dates.csv (which has a column 'acquisition_date' in the format '%Y%m%d')
          2. Converts these dates to numerical values (days since a reference)
          3. Computes the global union (sorted) of all dates across regions
          4. For each pixel time series (CSV file in a crop class folder), uses its region’s dates as observed times,
             and reinterpolates the spectral values onto the global timeline using Gaussian filtering.
          5. Normalizes spectral bands if requested.
        """
        self.root_dir = root_dir
        self.normalize = normalize
        self.data = []   # List of arrays of shape (T_global, C)
        self.labels = [] # Crop class labels (as integers)
        self.class_names = []  # Unique crop class names (e.g., "1", "2", ...)
        self.region_dates = {}  # Mapping from region to its dates (as np.array of floats)

        # First pass: For each region, read its dates.csv and store the dates.
        all_dates = []
        regions = sorted(os.listdir(root_dir))
        for region in tqdm(regions, desc="Processing regions"):
            region_path = os.path.join(root_dir, region)
            if not os.path.isdir(region_path):
                continue
            dates_path = os.path.join(region_path, "dates.csv")
            if os.path.exists(dates_path):
                df_dates = pd.read_csv(dates_path)
                df_dates['acquisition_date'] = pd.to_datetime(df_dates['acquisition_date'], format='%Y%m%d')
                region_date_vals = df_dates['acquisition_date'].apply(lambda d: d.toordinal()).values.astype(float)
                self.region_dates[region] = region_date_vals
                all_dates.append(region_date_vals)
            else:
                print(f"Warning: No dates.csv found in region {region}.")
        
        if len(all_dates) > 0:
            global_dates = np.unique(np.concatenate(all_dates))
            self.global_dates = np.sort(global_dates)
        else:
            raise ValueError("No dates found in any region.")
        
        # Now, for each region and each crop class folder, load each CSV file and reinterpolate onto the global timeline.
        for region in tqdm(regions, desc="Loading time series"):
            region_path = os.path.join(root_dir, region)
            if not os.path.isdir(region_path):
                continue
            if region in self.region_dates:
                region_date_vals = self.region_dates[region]
            else:
                continue
            items = sorted(os.listdir(region_path))
            for item in items:
                item_path = os.path.join(region_path, item)
                if os.path.isdir(item_path):
                    crop_class = item
                    if crop_class not in self.class_names:
                        self.class_names.append(crop_class)
                    label = self.class_names.index(crop_class)
                    files = sorted(os.listdir(item_path))
                    for file in tqdm(files, desc=f"Region {region} - Crop {crop_class}", leave=False):
                        if file.endswith('.csv'):
                            file_path = os.path.join(item_path, file)
                            try:
                                raw = np.loadtxt(file_path, delimiter=',', skiprows=1)
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")
                                continue
                            if raw.ndim == 1:
                                raw = raw.reshape(-1, raw.shape[0])
                            if raw.shape[1] < 2:
                                print(f"File {file_path} has insufficient columns.")
                                continue
                            values = raw[:, :-1]
                            mask = raw[:, -1]
                            T, C = values.shape
                            if T != len(region_date_vals):
                                print(f"Warning: {file_path} has {T} rows but region {region} dates has {len(region_date_vals)} rows.")
                            sigma = 7.0
                            T_global = len(self.global_dates)
                            interpolated = np.zeros((T_global, C))
                            for i, global_day in enumerate(self.global_dates):
                                diffs = region_date_vals - global_day
                                weights = np.exp(-0.5 * (diffs/sigma)**2) * mask
                                if weights.sum() > 0:
                                    interpolated[i] = (weights[:, None] * values).sum(axis=0) / weights.sum()
                                else:
                                    interpolated[i] = 0
                            self.data.append(interpolated)
                            self.labels.append(label)
                else:
                    continue
        
        try:
            self.data = np.array(self.data)
        except Exception as e:
            print("Error converting data to NumPy array:", e)
            raise e

        if self.normalize and self.data.size > 0:
            data_vals = self.data.reshape(-1, self.data.shape[-1])
            self.band_means = data_vals.mean(axis=0)
            self.band_stds = data_vals.std(axis=0) + 1e-8
            self.data = (self.data - self.band_means) / self.band_stds

    def get_global_dates(self):
        """Return the global timeline as pandas datetime objects."""
        return pd.to_datetime(self.global_dates, unit='D', origin='julian')

def train_test_split(dataset, test_fraction=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(dataset.data))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_data = dataset.data[train_idx]
    train_labels = np.array(dataset.labels)[train_idx]
    test_data = dataset.data[test_idx]
    test_labels = np.array(dataset.labels)[test_idx]
    
    train_set = SatelliteTimeSeriesDataset(dataset.root_dir, normalize=False)
    test_set = SatelliteTimeSeriesDataset(dataset.root_dir, normalize=False)
    train_set.data, train_set.labels, train_set.class_names = train_data, train_labels, dataset.class_names
    test_set.data, test_set.labels, test_set.class_names = test_data, test_labels, dataset.class_names
    if dataset.normalize:
        train_set.band_means, train_set.band_stds = dataset.band_means, dataset.band_stds
        test_set.band_means, test_set.band_stds = dataset.band_means, dataset.band_stds
        train_set.data = (train_set.data - train_set.band_means) / train_set.band_stds
        test_set.data = (test_set.data - test_set.band_means) / test_set.band_stds
    train_set.global_dates = dataset.global_dates
    test_set.global_dates = dataset.global_dates
    return train_set, test_set
