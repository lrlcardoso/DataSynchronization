"""
Signal processing utilities: resampling, normalization, correlation.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import correlate

# def resample_signal(signal, target_freq, time_col='Unix Time'):
#     """
#     Resample a signal DataFrame to the target frequency using linear interpolation.

#     Parameters:
#         signal (pd.DataFrame): Input data, must include a time column (default 'Unix Time').
#         orig_freq (float): Original sampling frequency (Hz).
#         target_freq (float): Target sampling frequency (Hz).
#         time_col (str): Name of the time column (default 'Unix Time').

#     Returns:
#         pd.DataFrame: Resampled DataFrame with same columns as input, at new frequency.
#     """
#     # Set up new uniformly spaced time axis
#     start_time = signal[time_col].iloc[0]
#     end_time = signal[time_col].iloc[-1]
#     num_samples = int(np.floor((end_time - start_time) * target_freq)) + 1
#     new_times = np.linspace(start_time, end_time, num_samples)

#     # Interpolate each column except time
#     resampled = {time_col: new_times}
#     for col in signal.columns:
#         if col == time_col:
#             continue
#         resampled[col] = np.interp(
#             new_times,
#             signal[time_col].values,
#             signal[col].values
#         )
#     resampled_df = pd.DataFrame(resampled)
#     return resampled_df

def position_to_acceleration(df, x_col, y_col):
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

    time_col='Unix Time'
    acc_col='Magnitude'

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


def design_lowpass_filter(cutoff, fs, order):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='low')
    return b, a

def design_highpass_filter(cutoff, fs, order):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='high')
    return b, a

def lowpass_filter_avoiding_gaps(df, fs, cutoff, order):
    b, a = design_lowpass_filter(cutoff, fs, order)
    padlen = 3 * max(len(a), len(b)) + 1

    filtered_data = df.copy()

    for col in df.columns:
        if col == "Unix Time":
            continue

        signal = df[col].values
        filtered_signal = np.full_like(signal, np.nan, dtype=float)

        valid_idx = signal != 0
        valid_signal = signal[valid_idx]
        if len(valid_signal) == 0:
            continue

        valid_time = np.where(valid_idx)[0] / fs
        time_diffs = np.diff(valid_time)
        gap_logical = np.concatenate([[True], time_diffs > 1.5 / fs, [True]])
        chunk_boundaries = np.where(gap_logical)[0]
        valid_positions = np.where(valid_idx)[0]

        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1]
            chunk = valid_signal[start:end]
            pos_start = valid_positions[start]
            pos_end = valid_positions[end - 1] + 1  # exclusive

            if len(chunk) >= padlen:
                try:
                    filtered_chunk = filtfilt(b, a, chunk)
                except Exception as e:
                    print(f"⚠️ Filter failed for column '{col}' chunk {i} (len={len(chunk)}): {e}")
                    filtered_chunk = chunk
            else:
                filtered_chunk = chunk  # fallback: keep original (will be interpolated)

            filtered_signal[pos_start:pos_end] = filtered_chunk

        # Interpolate across all NaNs (zero regions)
        nans = np.isnan(filtered_signal)
        not_nans = ~nans
        if np.sum(not_nans) >= 2:
            filtered_signal[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(not_nans),
                filtered_signal[not_nans]
            )
        else:
            # fallback: no interpolation possible
            filtered_signal[nans] = 0

        filtered_data[col] = filtered_signal

    return filtered_data

def highpass_filter(df, fs, cutoff, order):
    b, a = design_highpass_filter(cutoff, fs, order)
    filtered_df = df.copy()

    for col in df.columns:
        if col == "Unix Time":
            continue

        signal = df[col].values
        padlen = 3 * max(len(a), len(b)) + 1

        if len(signal) >= padlen:
            try:
                filtered_df[col] = filtfilt(b, a, signal)
            except Exception as e:
                print(f"⚠️ High-pass filter failed for column '{col}': {e}")
                filtered_df[col] = signal  # fallback
        else:
            print(f"⚠️ Skipping filtering for {col} (too short)")
            filtered_df[col] = signal

    return filtered_df


def lowpass_filter(df, fs, cutoff, order):
    """
    Apply low-pass Butterworth filter to all columns in the DataFrame
    except 'Unix Time'.

    Parameters:
    - df: pd.DataFrame, with 'Unix Time' and signal columns
    - fs: float, sampling frequency (e.g., 30 Hz)
    - cutoff: float, low-pass cutoff frequency (e.g., 1.0 Hz)
    - order: int, filter order (e.g., 2)

    Returns:
    - filtered_df: pd.DataFrame, same format as input but filtered
    """
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='low')
    filtered_df = df.copy()

    for col in df.columns:
        if col == "Unix Time":
            continue
        signal = df[col].values
        try:
            filtered_df[col] = filtfilt(b, a, signal)
        except ValueError as e:
            print(f"⚠️ Filtering failed for column '{col}': {e}")
            filtered_df[col] = signal  # fallback: unfiltered

    return filtered_df

# def resample_signal(df, target_freq):
#     if df.empty:
#         raise ValueError("Input DataFrame is empty.")
    
#     time_col = 'Unix Time'

#     # Define new time vector
#     start_time = df[time_col].iloc[0]
#     end_time = df[time_col].iloc[-1]
#     duration = end_time - start_time
#     num_samples = int(np.floor(duration * target_freq)) + 1
#     new_times = np.linspace(start_time, end_time, num_samples)

#     # Prepare output dictionary with interpolated columns
#     resampled = {time_col: new_times}
#     for col in df.columns:
#         if col == time_col:
#             continue
#         resampled[col] = np.interp(new_times, df[time_col].values, df[col].values)

#     return pd.DataFrame(resampled)

def resample_signal(df, target_freq, fill_missing_with_zero=False):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    time_col = "Unix Time"
    signal_cols = [col for col in df.columns if col != time_col]

    start_time = df[time_col].iloc[0]
    end_time = df[time_col].iloc[-1]
    duration = end_time - start_time

    t_new = np.arange(0, duration, 1 / target_freq)
    t_unix = start_time + t_new  # Absolute Unix timestamps

    if fill_missing_with_zero:
        # Zero-filling approach
        resampled_array = np.zeros((len(t_new), len(signal_cols)))
        index_map = np.round((df[time_col] - start_time) * target_freq).astype(int)

        for col_idx, col in enumerate(signal_cols):
            signal = df[col].values
            for i, idx in enumerate(index_map):
                if 0 <= idx < len(t_new):
                    resampled_array[idx, col_idx] = signal[i]

        resampled_df = pd.DataFrame(resampled_array, columns=signal_cols)
    else:
        # Interpolation approach
        resampled_dict = {time_col: t_unix}
        for col in signal_cols:
            resampled_dict[col] = np.interp(t_unix, df[time_col].values, df[col].values)
        resampled_df = pd.DataFrame(resampled_dict)
        return resampled_df

    # Final assembly for zero-fill case
    resampled_df.insert(0, time_col, t_unix)
    return resampled_df



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
