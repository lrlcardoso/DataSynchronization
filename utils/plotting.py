"""
Plotting functions for lag similarity and aligned signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import spectrogram

def plot_similarity_vs_lag(similarity_curve, fs, lag_range, label, output_dir, save=True):
    """
    Plots similarity (cross-correlation) curve as a function of lag (in seconds).

    Parameters:
        similarity_curve (np.array): Cross-correlation values.
        fs (float): Sampling frequency.
        lag_range (float): Maximum lag (in seconds).
        label (str): Identifier for the segment/camera.
        output_dir (str): Directory to save the plot.
        save (bool): Whether to save the plot as a PNG.
    """
    n = len(similarity_curve)
    lags = np.linspace(-lag_range, lag_range, n)

    plt.figure(figsize=(10, 4))
    plt.plot(lags, similarity_curve)
    plt.xlabel("Lag (s)")
    plt.ylabel("Normalized Correlation")
    plt.title(f"Similarity Curve vs. Lag ({label})")
    plt.grid(True)
    plt.tight_layout()
    if save:
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"{label}_similarity_vs_lag.png")
        plt.savefig(outpath, dpi=200)
        print(f"Saved similarity plot: {outpath}")
    plt.show()


def plot_aligned_signals(imu_norm, video_norm, lag_samples, fs, label, output_dir, save=False):
    """
    Plot IMU and video normalized signals with the optimal lag applied to video timestamps.
    """
    # Shift video time by lag (in seconds)
    shift_seconds = lag_samples / fs
    shifted_video = video_norm.copy()
    shifted_video['Unix Time'] += shift_seconds

    # Relative time axes
    imu_time = imu_norm['Unix Time'] - imu_norm['Unix Time'].iloc[0]
    video_time = shifted_video['Unix Time'] - imu_norm['Unix Time'].iloc[0]  # align to IMU start

    plt.figure(figsize=(12, 5))
    plt.plot(imu_time, imu_norm['Magnitude'], label="IMU", alpha=0.8)
    plt.plot(video_time, shifted_video['Magnitude'], label=f"Video (shifted by {shift_seconds:.3f} s)", alpha=0.8)
    plt.xlabel("Segment Time (s)")
    plt.ylabel("Normalized Magnitude")
    plt.title(f"Aligned Signals - {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{label}_aligned_signals.png")
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_debug(video_data_resampled, video_data_bandpass_filtered, markers, labels=None):
    """
    Plots a debug comparison of resampled vs bandpass-filtered video data.

    Parameters:
    - video_data_resampled: DataFrame with resampled signal (must include 'Unix Time')
    - video_data_bandpass_filtered: DataFrame with bandpass-filtered signal (must include 'Unix Time')
    - markers: str or tuple/list of two marker column names (e.g., '10_x' or ('10_x', '10_x'))
    - labels: Optional tuple/list of two strings for the legend
    """
    if isinstance(markers, str):
        markers = (markers, markers)
    elif isinstance(markers, (list, tuple)) and len(markers) == 1:
        markers = (markers[0], markers[0])
    elif not isinstance(markers, (list, tuple)) or len(markers) != 2:
        raise ValueError("`markers` must be a string or a tuple/list of exactly two column names.")

    label1 = labels[0] if labels and len(labels) > 0 else "Signal 1"
    label2 = labels[1] if labels and len(labels) > 1 else "Signal 2"

    plt.figure(figsize=(12, 5))

    # Resampled signal
    plt.plot(
        video_data_resampled["Unix Time"], video_data_resampled[markers[0]],
        label=label1, color='tab:orange', linewidth=1.5, alpha=0.8, marker='.', markersize=4
    )

    # Bandpass-filtered signal
    plt.plot(
        video_data_bandpass_filtered["Unix Time"], video_data_bandpass_filtered[markers[1]],
        label=label2, color='tab:green', linewidth=2, linestyle='-', alpha=0.9, marker='.', markersize=4
    )

    # Labels and formatting
    plt.xlabel("Unix Time (s)")
    plt.ylabel("Signal Value")
    plt.title(f"{markers[0]} vs {markers[1]} — {label1} vs {label2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_spectrograms(video_norm, fs=30):  # Set fs = your actual sampling rate
    """
    Plots spectrograms of all signal columns in video_norm.
    
    Parameters:
    - video_norm: pd.DataFrame with first column as time
    - fs: Sampling frequency in Hz (default = 100)
    """
    signal_cols = video_norm.columns[1:]  # skip time column
    time_vec = video_norm.iloc[:, 0].values
    time_vec = (time_vec - time_vec[0])  # convert Unix time to relative seconds
    
    for col in signal_cols:
        signal = video_norm[col].values

        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')  # dB scale
        plt.title(f"Spectrogram of {col}")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.colorbar(label="Power [dB]")
        plt.tight_layout()
        plt.show()