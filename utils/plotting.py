"""
Plotting functions for lag similarity and aligned signals.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_and_save_magnitude(data1, data2, camera=None, output_dir="Plots/", save=False, show=False):
    markers = ("Magnitude", "Magnitude")
    
    labels = ["Video", "IMU"]
    label1 = labels[0] if len(labels) > 0 else "Signal 1"
    label2 = labels[1] if len(labels) > 1 else "Signal 2"

    # Convert Unix Time to datetime for plotting
    x1 = pd.to_datetime(data1["Unix Time"], unit='s')
    x2 = pd.to_datetime(data2["Unix Time"], unit='s')

    plt.figure(figsize=(12, 5))

    # Plot first signal
    plt.plot(
        x1, data1[markers[0]],
        label=label1, color='tab:orange', linewidth=1.5, alpha=0.8, marker='.', markersize=4
    )

    # Plot second signal
    plt.plot(
        x2, data2[markers[1]],
        label=label2, color='tab:green', linewidth=2, linestyle='-', alpha=0.9, marker='.', markersize=4
    )

    # Labels and formatting
    plt.xlabel("Timestamp")
    plt.ylabel("Normalized Magnitude")
    plt.title(f"{label1} vs {label2} ({camera})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"Magnitude_Comparison_{camera}.png")
        plt.savefig(outpath, dpi=200)
        print(f"✅ Saved magnitude plot.")
    
    if show:
        plt.show()

    plt.close()

def plot_and_save_similarity(similarity_curve, lag_range, label, output_dir="Plots/", save=False, show=False):

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
        outpath = os.path.join(output_dir, f"Similarity_{label}.png")
        plt.savefig(outpath, dpi=200)
        print(f"✅ Saved similarity plot.")
    
    if show:
        plt.show()

    plt.close()

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

    plt.close()

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