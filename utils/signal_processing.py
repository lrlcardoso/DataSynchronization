"""
Signal processing utilities: resampling, normalization, correlation.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import correlate

def resample_signal(signal, target_freq, time_col='Unix Time'):
    """
    Resample a signal DataFrame to the target frequency using linear interpolation.

    Parameters:
        signal (pd.DataFrame): Input data, must include a time column (default 'Unix Time').
        orig_freq (float): Original sampling frequency (Hz).
        target_freq (float): Target sampling frequency (Hz).
        time_col (str): Name of the time column (default 'Unix Time').

    Returns:
        pd.DataFrame: Resampled DataFrame with same columns as input, at new frequency.
    """
    # Set up new uniformly spaced time axis
    start_time = signal[time_col].iloc[0]
    end_time = signal[time_col].iloc[-1]
    num_samples = int(np.floor((end_time - start_time) * target_freq)) + 1
    new_times = np.linspace(start_time, end_time, num_samples)

    # Interpolate each column except time
    resampled = {time_col: new_times}
    for col in signal.columns:
        if col == time_col:
            continue
        resampled[col] = np.interp(
            new_times,
            signal[time_col].values,
            signal[col].values
        )
    resampled_df = pd.DataFrame(resampled)
    return resampled_df

def position_to_acceleration(df, time_col='Unix Time', x_col='11_x', y_col='11_y', acc_col='Magnitude'):
    """
    Given a DataFrame with time and x/y position, compute acceleration magnitude.
    
    Parameters:
        df (pd.DataFrame): Must include time_col, x_col, y_col.
        time_col (str): Time column.
        x_col (str): X position column.
        y_col (str): Y position column.
        acc_col (str): Output acceleration magnitude column name.
        
    Returns:
        pd.DataFrame: [time_col, acc_col]
    """
    t = df[time_col].values
    x = df[x_col].values
    y = df[y_col].values

    # First derivative: velocity
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)

    # Second derivative: acceleration
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    acc_mag = np.sqrt(ax**2 + ay**2)
    out = pd.DataFrame({time_col: t, acc_col: acc_mag})
    return out

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

def highpass_filter(series, freq, cutoff=0.1, order=2):
    """
    Apply a Butterworth high-pass filter to the magnitude column of the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame (must contain mag_col).
        freq (float): Sampling frequency (Hz).
        cutoff (float): Cutoff frequency for high-pass filter (Hz).
        order (int): Filter order.
        mag_col (str): Name of the magnitude column to filter.

    Returns:
        pd.DataFrame: Copy of df with filtered mag_col.
    """
    nyq = 0.5 * freq
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, series.values)

def lowpass_filter(series, freq, cutoff=10.0, order=2):
    """
    Apply a Butterworth low-pass filter to the magnitude column of the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame (must contain mag_col).
        freq (float): Sampling frequency (Hz).
        cutoff (float): Cutoff frequency for low-pass filter (Hz).
        order (int): Filter order.
        mag_col (str): Name of the magnitude column to filter.

    Returns:
        pd.DataFrame: Copy of df with filtered mag_col.
    """
    nyq = 0.5 * freq
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, series.values)

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
    Computes cross-correlation between IMU and video magnitude signals.
    Shifts video signal to align with IMU (IMU is reference).

    Parameters:
        imu_norm (pd.DataFrame): IMU signal with uniform sampling.
        video_norm (pd.DataFrame): Video signal with uniform sampling.
        fs (float): Sampling frequency (Hz).
        lag_range (float): Maximum absolute lag to consider (seconds).
        mag_col (str): Column with magnitude values.

    Returns:
        lag_samples (int): Lag (in samples) for max correlation (positive = video lags behind IMU).
        max_corr (float): Maximum correlation value.
        similarity_curve (np.array): Cross-correlation curve for valid lags.
    """
    x = imu_norm[mag_col].values
    y = video_norm[mag_col].values
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    corr = correlate(x, y, mode='full')
    lags = np.arange(-len(x)+1, len(x))

    # Normalize
    norm_factor = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    corr = corr / norm_factor
    
    # Limit to desired lag window (in samples)
    max_lag_samples = int(lag_range * fs)
    center = len(corr) // 2
    valid_range = (center - max_lag_samples, center + max_lag_samples + 1)
    valid_corr = corr[valid_range[0]:valid_range[1]]
    valid_lags = lags[valid_range[0]:valid_range[1]]

    # Find lag of maximum correlation
    max_idx = np.argmax(valid_corr)
    lag_samples = valid_lags[max_idx]
    max_corr = valid_corr[max_idx]
    
    return lag_samples, max_corr, valid_corr
