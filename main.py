"""
==============================================================================
Title:          Signal Synchronization Pipeline (Folder-Based)
Description:    Aligns and synchronizes IMU and video marker data for all
                segment folders found inside the input directory.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-25
Version:        1.1
==============================================================================
"""

import os
import pandas as pd
from config import *
import matplotlib.pyplot as plt
from utils.file_io import load_imu_data, load_video_marker_data, determine_wrist_side
from utils.signal_processing import resample_signal, compute_magnitude, highpass_filter, normalize_signal, compute_cross_correlation
from utils.plotting import plot_similarity_vs_lag, plot_aligned_signals
from utils.data_merge import merge_and_save_csv
from config import IMU_DATA_DIR, VIDEO_MARKER_DIR, IMU_FREQ, VIDEO_FREQ, COMMON_FREQ

def plot_normalized_signals(imu_norm, video_norm, label_imu='IMU', label_video='Video'):
    """
    Plots normalized IMU and video magnitudes on the same time axis.

    Parameters:
        imu_norm (pd.DataFrame): Normalized IMU DataFrame.
        video_norm (pd.DataFrame): Normalized video DataFrame.
        label_imu (str): Label for IMU.
        label_video (str): Label for video.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(imu_norm['Unix Time'], imu_norm['Magnitude'], label=label_imu, alpha=0.8)
    plt.plot(video_norm['Unix Time'], video_norm['Magnitude'], label=label_video, alpha=0.8)
    plt.xlabel('Unix Time (s)')
    plt.ylabel('Normalized Magnitude')
    plt.title('Normalized IMU vs Video Magnitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def extract_central_window(df, window_sec=2.0, time_col='Unix Time'):
    """
    Returns a DataFrame with only the central window of given seconds.
    """
    center_time = 0.5 * (df[time_col].iloc[0] + df[time_col].iloc[-1])
    half_win = window_sec / 2
    mask = (df[time_col] >= center_time - half_win) & (df[time_col] <= center_time + half_win)
    return df[mask].reset_index(drop=True)


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
    out[mag_col] = out[mag_col].rolling(window=window, center=True, min_periods=window).mean()

    return out

def main():
    input_root = r"C:\Users\s4659771\Documents\MyTurn_Project\Data\Processed\P01\Session2_20250210\Video\VR\Camera1\Segments"  
    segment_folders = [os.path.join(input_root, d) for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    for seg_path in segment_folders:
        print(f"\nProcessing segment folder: {os.path.basename(seg_path)}")
        # Find CSV file (assume only one .csv per folder)
        csv_files = [f for f in os.listdir(seg_path) if f.endswith('.csv')]
        if not csv_files:
            print("  No CSV file found in this folder, skipping.")
            continue
        csv_path = os.path.join(seg_path, csv_files[0])
        df = pd.read_csv(csv_path)
        if df.empty:
            print("  CSV is empty, skipping.")
            continue
        
        # Get UNIX start and end from first/last row
        seg_start = df.iloc[0]['Unix Time']   # Adjust column name as needed
        seg_end = df.iloc[-1]['Unix Time']
        seg_interval = (seg_start - TIME_MARGIN, seg_end + TIME_MARGIN)
        
        # Determine camera/wrist side from folder or file name
        camera_label = os.path.basename(os.path.dirname(os.path.dirname(seg_path)))
        # wrist_side = determine_wrist_side(camera_label)
        wrist_side = "LH" # this is temporary until I develop the determine_wrist_side method
        
        # Load IMU and video marker data for interval
        imu_data = load_imu_data(IMU_DATA_DIR, wrist_side, seg_interval)
        video_data = df[['Unix Time', '10_x', '10_y']] #load_video_marker_data(VIDEO_MARKER_DIR, camera_label, seg_interval)

        # Magnitude
        imu_mag = compute_magnitude(imu_data)
        video_mag = compute_magnitude(video_data)
        
        # Resample
        imu_rs = resample_signal(imu_mag, COMMON_FREQ)
        video_rs = resample_signal(video_mag, COMMON_FREQ)

        # Remove the gravity component from the imu data
        imu_mag_filtered = highpass_filter(imu_rs, freq=COMMON_FREQ)
        video_mag_filtered = highpass_filter(video_rs, freq=COMMON_FREQ)

        window_size = 50  # Choose a window (e.g., 5–21 depending on your signal)
        imu_smooth = smooth_signal(imu_mag_filtered, window=window_size)

        # window_size = 500
        # video_smooth = smooth_signal(video_rs, window=window_size)
        
        # Normalize
        NORM_METHOD = "zscore"
        imu_norm = normalize_signal(imu_smooth, NORM_METHOD)
        video_norm = normalize_signal(video_mag_filtered, NORM_METHOD)

        plot_normalized_signals(imu_norm, video_norm)

#         imu_center = extract_central_window(imu_norm, window_sec=2.0)
#         video_center = extract_central_window(video_norm, window_sec=2.0)

#         # Now compute cross-correlation just on these:
#         lag_samples, max_corr, similarity_curve = compute_cross_correlation(
#             imu_center, video_center, COMMON_FREQ, LAG_RANGE
# )
        
#         # # Cross-correlate
#         # lag_samples, max_corr, similarity_curve = compute_cross_correlation(
#         #     imu_norm, video_norm, COMMON_FREQ, LAG_RANGE
#         # )
#         print(f"  Optimal lag (s): {lag_samples/COMMON_FREQ:.3f}, Correlation: {max_corr:.3f}")
        
#         # Plots
#         if SHOW_PLOTS or SAVE_PLOTS:
#             plot_similarity_vs_lag(similarity_curve, COMMON_FREQ, LAG_RANGE, camera_label, OUTPUT_DIR, save=False)
#             # plot_aligned_signals(imu_norm, video_norm, lag_samples, COMMON_FREQ, camera_label, OUTPUT_DIR, save=SAVE_PLOTS)
        
        # # Merge and save CSV
        # if SAVE_CSV:
        #     merge_and_save_csv(imu_rs, video_rs, lag_samples, COMMON_FREQ, camera_label, OUTPUT_DIR)

if __name__ == "__main__":
    main()
