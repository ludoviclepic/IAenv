import os
import numpy as np
import pandas as pd
from glob import glob
import collections.abc
import re

import torch
from torch.nn import functional as F


def read_tile_dates(dates_csv_path: str) -> pd.DataFrame:
    """
    Reads the tile's dates.csv with shape (T,1) or (T,...) where each row is an acquisition date.
    Returns a DataFrame of shape (T,) with a 'acquisition_date' column (or whatever you like).
    You can store them as strings or parse them to datetime objects, as needed.
    """
    df_dates = pd.read_csv(dates_csv_path)
    # For example, if there's only one column 'acquisition_date' in format YYYYMMDD:
    df_dates['date_dt'] = pd.to_datetime(df_dates['acquisition_date'], format='%Y%m%d')
    return df_dates

def read_pixel_csv(pixel_csv_path: str) -> pd.DataFrame:
    """
    Reads one pixel CSV, which has T rows (same T as tile's dates.csv).
    Columns: B1..B9, Flag. No date column. The row i matches the i-th date in dates.csv.
    """
    df = pd.read_csv(pixel_csv_path)
    return df  # shape (T, 10) e.g. columns B1..B9, Flag

def load_tile_data(tile_path: str):
    """
    For a single tile folder:
      - read tile's dates.csv => T lines
      - for each crop folder (0..15), read each pixel CSV => also T lines
      - filter out Flag != 0 (optional: set to NaN, or remove them, etc.)
      - produce arrays of shape (T, 9) for each valid pixel
      - return X_list, y_list
    """
    # 1) Read the tile's date info
    tile_dates_path = os.path.join(tile_path, "dates.csv")
    df_dates = read_tile_dates(tile_dates_path)
    T = len(df_dates)  # number of acquisitions for this tile

    X_list = []
    y_list = []

    # 2) For each crop label (0..15), read all pixel CSVs in that folder
    for class_str in map(str, range(16)):
        crop_folder = os.path.join(tile_path, class_str)
        if not os.path.isdir(crop_folder):
            continue

        pixel_csv_files = glob(os.path.join(crop_folder, "*.csv"))
        crop_label = int(class_str)

        for csv_path in pixel_csv_files:
            df_pixel = read_pixel_csv(csv_path)
            # Expect df_pixel.shape[0] == T

            if len(df_pixel) != T:
                # If there's a mismatch in row count, you must handle it:
                # e.g. skip or raise an error, or see how your data is structured
                continue

            # 3) Filter out invalid flags by setting reflectances to NaN or 0
            #    You do not want to drop rows because you'd lose alignment with the date index.
            bad_flag_idx = df_pixel['Flag'] != 0
            for band_col in [f"B{i}" for i in range(1,10)]:
                df_pixel.loc[bad_flag_idx, band_col] = np.nan  # or 0

            # 4) Extract the reflectance columns as a (T, 9) array
            #    (assuming columns are named B1..B9)
            reflectance_cols = [f"B{i}" for i in range(1,10)]
            arr = df_pixel[reflectance_cols].to_numpy(dtype=np.float32)  # shape (T,9)

            # Optional: you can do some interpolation for the NaNs
            # e.g. fill forward/backward or leave them as NaN
            # We'll just leave them as NaN for demonstration

            X_list.append(arr)       # shape (T,9)
            y_list.append(crop_label)

    return X_list, y_list

def load_dataset(root_dir: str):
    """
    Walks over all tiles in root_dir, loads their data, returns X,y as lists (or arrays).
    """
    from glob import glob
    tile_paths = [p for p in glob(os.path.join(root_dir, "*")) if os.path.isdir(p)]

    all_X = []
    all_y = []
    for tile_path in tile_paths:
        X_list, y_list = load_tile_data(tile_path)
        all_X.extend(X_list)
        all_y.extend(y_list)

    # Convert to arrays or keep as lists if variable length:
    X = np.array(all_X, dtype=object)  # if T varies among tiles, store dtype=object
    y = np.array(all_y, dtype=np.int64)
    return X, y


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def pad_collate(batch, pad_value=0):
    # Utility function to be used as collate_fn for the PyTorch dataloader
    # to handle sequences of varying length.
    # Sequences are padded with zeros by default.
    #
    # Modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}

    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))

    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))