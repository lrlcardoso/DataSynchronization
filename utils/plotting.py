"""
Plotting functions for lag similarity and aligned signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

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


def plot_aligned_signals(sig1, sig2, lag_samples, fs, seg, output_dir, save=False):
    """
    Overlays IMU and video marker signals after lag alignment.
    """
    # TODO: Implement plotting with matplotlib
    pass
