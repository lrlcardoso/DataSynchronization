"""
==============================================================================
Title:          Signal Processing Utilities
Description:    Provides functions for filtering, normalization, correlation, 
                resampling, signal alignment, acceleration estimation, and 
                energy transformation used in synchronizing IMU and video data.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-25
Version:        1.0
==============================================================================
Usage:
    from utils.signal_processing import (
        resample_signal, normalize_signal, compute_cross_correlation,
        compute_magnitude, align_signals, position_to_acceleration,
        highpass_filter, lowpass_filter, smooth_signal, teager_kaiser_energy,
        apply_lag
    )

Dependencies:
    - Python >= 3.x
    - Required libraries: numpy, pandas, scipy

Notes:
    - All filters handle non-continuous (NaN-separated) signal segments robustly.
    - Acceleration is derived from position using second-order gradient.

Changelog:
    - v1.0: [2025-04-25] Initial release
==============================================================================
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import correlate

def apply_lag(data, lag_samples, fs):

    unix_col = data.columns[0]
    seg_col = data.columns[1]
    other_cols = data.columns[2:]

    n_samples = len(data)
    lag_sec = lag_samples / fs

    # Apply lag to Unix Time only
    start_unix = data[unix_col].iloc[0] - lag_sec
    unix_time = np.floor((start_unix + np.arange(n_samples) / fs) * 1000) / 1000

    # Build new DataFrame
    data_lagged = pd.DataFrame({
        unix_col: unix_time,
        seg_col: data[seg_col].values
    })

    for col in other_cols:
        data_lagged[col] = data[col].values

    return data_lagged

def smooth_signal(df, window=5, mag_col="Magnitude"):
    """
    Applies a moving average (rolling mean) to the magnitude column.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'Magnitude' column.
        window (int): Window size (in samples).
        mag_col (str): Name of the column to smooth.

    Returns:
        pd.DataFrame: DataFrame with the same columns and smoothed magnitude.
    """
    out = df.copy()
    out[mag_col] = out[mag_col].rolling(window=window, center=True, min_periods=1).mean()

    return out

def teager_kaiser_energy(data):
    """
    Apply TKEO to signal columns, preserving time column.
    Works with both NumPy arrays and Pandas DataFrames.
    Assumes time is in the first column.
    """
    if isinstance(data, pd.DataFrame):
        time = data.iloc[:, [0]]
        signal = data.iloc[:, 1:].values
        tkeo = np.zeros_like(signal)
        tkeo[1:-1] = signal[1:-1]**2 - signal[:-2] * signal[2:]
        tkeo_df = pd.DataFrame(tkeo, columns=data.columns[1:], index=data.index)
        return pd.concat([time, tkeo_df], axis=1)

    else:  # assume NumPy
        time = data[:, [0]]
        signal = data[:, 1:]
        tkeo = np.zeros_like(signal)
        tkeo[1:-1] = signal[1:-1]**2 - signal[:-2] * signal[2:]
        return np.hstack((time, tkeo))

def position_to_acceleration(df, x_col, y_col):
    """
    Compute acceleration magnitude from position data, skipping over NaN regions.

    Parameters:
        df (pd.DataFrame): Must include 'Unix Time', x_col, and y_col.
        x_col (str): X position column name.
        y_col (str): Y position column name.

    Returns:
        pd.DataFrame: DataFrame with ['Unix Time', 'Magnitude'] where acceleration
                      is computed only for valid segments (others remain NaN).
    """
    time_col = 'Unix Time'
    acc_col = 'Magnitude'

    # Mask where data is valid
    valid_mask = df[[x_col, y_col]].notna().all(axis=1)

    # Pre-fill with NaNs
    acc_mag = np.full(len(df), np.nan)

    if valid_mask.sum() >= 5:  # At least enough points for gradient to be meaningful
        # Extract valid slices
        t_valid = df[time_col][valid_mask].values
        x_valid = df[x_col][valid_mask].values
        y_valid = df[y_col][valid_mask].values

        # Compute acceleration
        vx = np.gradient(x_valid, t_valid)
        vy = np.gradient(y_valid, t_valid)
        ax = np.gradient(vx, t_valid)
        ay = np.gradient(vy, t_valid)
        acc_valid = np.sqrt(ax**2 + ay**2)

        # Insert computed values into the full output array
        acc_mag[valid_mask.values] = acc_valid

    return pd.DataFrame({time_col: df[time_col], acc_col: acc_mag})

def compute_magnitude(df, time_col='Unix Time', mag_col='Magnitude'):
    """
    Compute vector magnitude using all columns except the time column.

    Parameters:
        df (pd.DataFrame): Input DataFrame. First column should be time.
        time_col (str): Name of the time column.
        mag_col (str): Name of the output magnitude column.

    Returns:
        pd.DataFrame: DataFrame with [time_col, mag_col].
    """
    spatial_cols = [col for col in df.columns if col != time_col]
    magnitude = np.linalg.norm(df[spatial_cols].values, axis=1)
    out = pd.DataFrame({time_col: df[time_col], mag_col: magnitude})
    return out

def design_lowpass_filter(cutoff, fs, order):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='low')
    return b, a

def design_highpass_filter(cutoff, fs, order):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='high')
    return b, a

def highpass_filter(df, fs, cutoff, order):
    """
    Apply high-pass Butterworth filter to valid (non-NaN) segments of each signal column.
    Segments too short for filtering are replaced with NaN.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Unix Time' and signal columns.
        fs (float): Sampling frequency.
        cutoff (float): High-pass cutoff frequency.
        order (int): Filter order.

    Returns:
        pd.DataFrame: Filtered DataFrame with same structure as input.
    """
    b, a = design_highpass_filter(cutoff, fs, order)
    padlen = 3 * max(len(a), len(b)) + 1
    filtered_df = df.copy()

    for col in df.columns:
        if col in ("Unix Time", "Segment Time") or "_conf" in col:
            continue

        signal = df[col]
        if signal.isnull().all():
            # print(f"Column '{col}' contains only NaN values and cannot be filtered.")
            continue

        # Initialize result with NaNs
        filtered_signal = pd.Series(np.nan, index=signal.index)

        # Find valid (non-NaN) continuous segments
        is_valid = signal.notna()
        valid_groups = (is_valid != is_valid.shift()).cumsum()[is_valid]

        for _, group_idx in valid_groups.groupby(valid_groups):
            idx = group_idx.index
            segment = signal.loc[idx]

            if len(segment) >= padlen:
                try:
                    filtered_segment = filtfilt(b, a, segment)
                    filtered_signal.loc[idx] = filtered_segment
                except Exception as e:
                    print(f"⚠️ High-pass filter failed for segment in '{col}': {e}")
                    # Already NaN
            else:
                # Force fill short segment with NaN
                filtered_signal.loc[idx] = np.nan

        filtered_df[col] = filtered_signal

    return filtered_df

def lowpass_filter(df, fs, cutoff, order):
    """
    Apply low-pass Butterworth filter to valid (non-NaN) segments of the signal.
    Segments too short for filtering are filled with NaN.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Unix Time' and signal columns.
        fs (float): Sampling frequency.
        cutoff (float): Low-pass cutoff frequency.
        order (int): Filter order.

    Returns:
        pd.DataFrame: Filtered DataFrame with same structure.
    """
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='low')
    padlen = 3 * max(len(a), len(b)) + 1
    filtered_df = df.copy()

    for col in df.columns:
        if col in ("Unix Time", "Segment Time") or "_conf" in col:
            continue

        signal = df[col]
        if signal.isnull().all():
            # print(f"Column '{col}' contains only NaN values and cannot be filtered.")
            continue
        
        # Start with full NaN array
        filtered_signal = pd.Series(np.nan, index=signal.index)

        # Identify valid (non-NaN) chunks
        is_valid = signal.notna()
        valid_groups = (is_valid != is_valid.shift()).cumsum()[is_valid]

        for _, group_idx in valid_groups.groupby(valid_groups):
            idx = group_idx.index
            segment = signal.loc[idx]

            if len(segment) >= padlen:
                try:
                    filtered_segment = filtfilt(b, a, segment)
                    filtered_signal.loc[idx] = filtered_segment
                except Exception as e:
                    print(f"⚠️ Filtering failed for segment in '{col}': {e}")
                    # Leave as NaN (already is)
            else:
                # Forcefully replace too-short segments with NaN
                filtered_signal.loc[idx] = np.nan

        filtered_df[col] = filtered_signal

    return filtered_df

def resample_signal(df, target_freq, fill_missing_with_nan):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    time_col = "Unix Time"
    
    # Identify numeric signal columns and non-numeric extra columns
    signal_cols = [col for col in df.columns if col != time_col and df[col].dtype in [np.float64, np.float32, np.int64]]
    extra_cols = [col for col in df.columns if col not in [time_col] + signal_cols]

    # Time setup
    start_time = df[time_col].iloc[0]
    end_time = df[time_col].iloc[-1]
    n_samples = int(np.round((end_time - start_time) * target_freq)) + 1

    t_new = np.arange(n_samples) / target_freq
    t_unix = np.floor((start_time + t_new) * 1000) / 1000

    if fill_missing_with_nan:
        # === NaN-filling for numeric signals ===
        resampled_array = np.full((len(t_new), len(signal_cols)), np.nan)
        index_map = np.round((df[time_col] - start_time) * target_freq).astype(int)

        for col_idx, col in enumerate(signal_cols):
            signal = df[col].values
            for i, idx in enumerate(index_map):
                if 0 <= idx < len(t_new):
                    val = signal[i]
                    resampled_array[idx, col_idx] = np.nan if val == 0 else val

        # Create DataFrame with signal columns
        resampled_df = pd.DataFrame(resampled_array, columns=signal_cols)

        # Insert Unix Time at column 0
        resampled_df.insert(0, time_col, t_unix)

        # Insert each extra column right after Unix Time
        for i, col in enumerate(extra_cols):
            extra_values = [None] * len(t_unix)
            for j, idx in enumerate(index_map):
                if 0 <= idx < len(t_unix):
                    extra_values[idx] = df[col].iloc[j]
            resampled_df.insert(1 + i, col, extra_values)

        return resampled_df

    else:
        # === Interpolation for numeric signals ===
        resampled_dict = {time_col: t_unix}
        for col in signal_cols:
            resampled_dict[col] = np.interp(t_unix, df[time_col].values, df[col].values)

        resampled_df = pd.DataFrame(resampled_dict)

        # Insert extra columns after Unix Time
        for i, col in enumerate(extra_cols):
            nearest_idx = np.searchsorted(df[time_col].values, t_unix, side='left')
            nearest_idx = np.clip(nearest_idx, 0, len(df) - 1)
            values = df[col].values[nearest_idx]
            resampled_df.insert(1 + i, col, values)

        return resampled_df

    
def align_signals(matrix1, matrix2, method='nearest'):
    df1 = pd.DataFrame(matrix1, columns=["Unix Time", "Magnitude"])
    df2 = pd.DataFrame(matrix2, columns=["Unix Time", "Magnitude"])

    df2_indexed = df2.set_index("Unix Time")

    if method == 'nearest':
        aligned2 = df2_indexed.reindex(df1["Unix Time"], method="nearest").reset_index()

    elif method == 'interp':
        # Interpolation requires datetime index
        union_times = df2_indexed.index.union(df1["Unix Time"]).sort_values()
        df2_union = df2_indexed.reindex(union_times)

        df2_union.index = pd.to_datetime(df2_union.index, unit='s')
        df1_times_dt = pd.to_datetime(df1["Unix Time"], unit='s')

        interpolated = df2_union.interpolate(method="time")
        aligned2 = interpolated.loc[df1_times_dt].reset_index()
        aligned2.rename(columns={"index": "Unix Time"}, inplace=True)
        aligned2["Unix Time"] = df1["Unix Time"].values  # restore original float times

    else:
        raise ValueError("method must be 'nearest' or 'interp'")

    aligned_matrix2 = pd.DataFrame(aligned2, columns=["Unix Time", "Magnitude"])
    return aligned_matrix2

def normalize_signal(df, method="zscore", mag_col="Magnitude"):
    """
    Normalize the magnitude column of the signal DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame (must contain mag_col).
        method (str): Normalization method, "zscore" or "minmax".
        mag_col (str): Name of the magnitude column to normalize.

    Returns:
        pd.DataFrame: Copy of df with normalized mag_col.
    """
    out = df.copy()
    if method == "zscore":
        mean = out[mag_col].mean()
        std = out[mag_col].std()
        out[mag_col] = (out[mag_col] - mean) / std
    elif method == "minmax":
        min_val = out[mag_col].min()
        max_val = out[mag_col].max()
        out[mag_col] = (out[mag_col] - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return out

def compute_cross_correlation(imu_norm, video_norm, fs, lag_range, mag_col="Magnitude"):
    """
    Computes cross-correlation between IMU and video signals, using only overlapping,
    non-NaN samples.

    Returns:
        lag_samples (int): Optimal lag (video relative to IMU, in samples).
        max_corr (float): Maximum normalized correlation.
        similarity_curve (np.ndarray): Cross-correlation curve within lag_range.
    """
    x = imu_norm[mag_col].values
    y = video_norm[mag_col].values

    # Align full length
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # Mask valid overlapping region
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    if not np.any(valid_mask):
        raise ValueError("No overlapping valid samples found.")

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) < 3:
        raise ValueError("Not enough valid data for cross-correlation.")

    # Compute full cross-correlation (y is shifted relative to x)
    corr = correlate(y_valid, x_valid, mode='full')
    lags = np.arange(-len(x_valid) + 1, len(x_valid))

    # Normalize
    norm = np.sqrt(np.sum(x_valid ** 2) * np.sum(y_valid ** 2))
    if norm == 0:
        raise ValueError("Cannot normalize correlation (zero energy).")
    corr /= norm

    # Limit to desired lag window
    max_lag_samples = int(lag_range * fs)
    center = len(corr) // 2
    start = max(center - max_lag_samples, 0)
    end = min(center + max_lag_samples + 1, len(corr))

    valid_corr = corr[start:end]
    valid_lags = lags[start:end]

    # Find best lag
    max_idx = np.argmax(valid_corr)
    lag_samples = valid_lags[max_idx]
    max_corr = valid_corr[max_idx]

    return lag_samples, max_corr, valid_corr