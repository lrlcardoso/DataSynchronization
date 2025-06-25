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
import time
import pandas as pd
import numpy as np
from config import *
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import zscore
from utils.file_utils import load_imu_data, get_videos_info, load_video_marker_data, determine_wrist_side
from utils.signal_processing import lowpass_filter_avoiding_gaps, resample_signal, compute_magnitude, highpass_filter, normalize_signal, lowpass_filter, position_to_acceleration, compute_cross_correlation
from utils.plotting import plot_debug, plot_similarity_vs_lag, plot_aligned_signals
from utils.data_merge import merge_and_save_csv
from config import IMU_DATA_DIR, LAG_RANGE, VIDEO_MARKER_DIR, IMU_FREQ, VIDEO_FREQ, FILTER_LOW_CUT, FILTER_HIGH_CUT, FILTER_ORDER, ROOT_DIR, SELECTED_PATIENTS, SELECTED_SESSIONS, SELECTED_SUBFOLDERS, SELECTED_SEGMENTS

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


def run_sync(video_path, patient, session, affected_side):

    # 0 - Status
    # 1 - Load all necessary data and info
    #   - video markers csv
    #   - camera side
    #   - imu csv (trimmed to the segment with margins)
    # 2 - Preprocessing
    #   - Video:
    #       - Add zero to every gap
    #       - Filter video data (avoiding zero or abrupt changes)
    #   - IMU:
    #       - Resample to 100Hz to make sure there is no gap
    #       - Filter imu data 
    # 3 - Compute magnitudes
    # 4 - Resample to a common freq (30Hz)
    # 5 - Smooth signals
    # 6 - Drop NA
    # 7 - Normalize
    # 8 - Cross-correlate
    # 9 - Plot and generate sync CSVs

    print("="*100)
    print(f"📝 Processing: {patient} ({affected_side}) | {session}")
    print("="*100)

    seg_dir = os.path.join(video_path, "Camera1", "Segments")

    all_seg_names = natsorted(os.listdir(seg_dir))

    if SELECTED_SEGMENTS is None:
        seg_names = all_seg_names
    else:
        seg_names = [all_seg_names[i] for i in SELECTED_SEGMENTS if i < len(all_seg_names)]

    for seg_name in seg_names:
        if "static" in seg_name.lower():
            print(f"Skipped segment {seg_name}.")
            continue  # Skip 'static' segments

        # Find in which camera we have more markers visible
        if affected_side == "R" or affected_side == "B":
            marker = "11_"
        elif affected_side == "L":
            marker = "10_"
        else:
            raise ValueError(f"Unknown affected_side: {affected_side}")
        
        best_camera = None
        max_visible_points = -1
        best_df = None

        for camera_name in ["Camera1", "Camera2"]:
            camera_path = os.path.join(video_path, camera_name, "Segments", seg_name)
            if not os.path.isdir(camera_path):
                continue  # Skip if camera folder is missing

            csv_files = [f for f in os.listdir(camera_path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found in folder: {camera_path}")
            if len(csv_files) > 1:
                raise RuntimeError(f"Multiple CSV files found in folder: {camera_path} → {csv_files}")

            csv_path = os.path.join(camera_path, csv_files[0])
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError(f"CSV file is empty: {csv_path}")

            x_col, y_col = f"{marker}x", f"{marker}y"
            if x_col not in df.columns or y_col not in df.columns:
                continue  # Skip if marker columns are missing

            visible_count = ((df[x_col] != 0) & (df[y_col] != 0)).sum()

            if visible_count > max_visible_points:
                max_visible_points = visible_count
                best_camera = camera_name
                best_df = df

        if best_camera is None or best_df is None:
            raise ValueError(f"No valid camera data found for segment: {seg_name}")
    
        print()
        print(f"📂 Segment {seg_name} ({best_camera})")
        print("-" * 100)

        # 1 - Load all necessary data and info
        # Video data
        video_data = best_df[['Unix Time', f"{marker}x", f"{marker}y"]].copy()
        video_data[f"{marker}y"] = -video_data[f"{marker}y"]
        # IMU data
        # Get UNIX start and end from first/last row
        seg_start = video_data.iloc[0]['Unix Time']
        seg_end = video_data.iloc[-1]['Unix Time']
        seg_interval = (seg_start - TIME_MARGIN, seg_end + TIME_MARGIN)
        imu_path = os.path.abspath(os.path.join(os.path.dirname(video_path), "..", "WMORE"))
        imu_data = load_imu_data(imu_path, affected_side, seg_interval)
        imu_data['ax'] = -imu_data['ax']

        # 2 - Preprocessing
        # Resample video_data to make sure thee is no gaps. Also add 0 to every gap to facilitate visualization
        video_data_resampled = resample_signal(video_data, target_freq=VIDEO_FREQ, fill_missing_with_zero=True)

        # Apply a lowpass filter to reduce noise, but do it avoiding the chunks in which the signal is zero (bad detection)
        video_data_lowpass_filtered = lowpass_filter_avoiding_gaps(video_data_resampled, fs=VIDEO_FREQ, cutoff=FILTER_HIGH_CUT, order=FILTER_ORDER)

        # Uncomment to plot (for debugging)
        # plot_debug(video_data_resampled, video_data_lowpass_filtered, f"{marker}y", labels=["Video - resampled", "Video - filtered"])

        # Apply a highpass filter to match to the IMU that must be highpassed filtered to get rid of the gravity effect
        video_data_bandpass_filtered = highpass_filter(video_data_lowpass_filtered, fs=VIDEO_FREQ, cutoff=FILTER_LOW_CUT, order=FILTER_ORDER)

        # Resample to IMU data to make sure there is no gap
        imu_data_resampled = resample_signal(imu_data, IMU_FREQ, fill_missing_with_zero=False)

        # Apply a lowpass filter to reduce noise, matching the video_data
        imu_data_lowpass_filtered = lowpass_filter(imu_data_resampled, fs=IMU_FREQ, cutoff=FILTER_HIGH_CUT, order=FILTER_ORDER)

        # Uncomment to plot (for debugging)
        # plot_debug(imu_data_resampled, imu_data_lowpass_filtered, "ax", labels=["IMU - resampled", "IMU - filtered"])

        # Apply a highpass filter to remove gravity
        imu_data_bandpass_filtered = highpass_filter(imu_data_lowpass_filtered, fs=IMU_FREQ, cutoff=FILTER_HIGH_CUT, order=FILTER_ORDER)

        # 3 - Compute magnitudes
        video_mag = position_to_acceleration(video_data_bandpass_filtered, f"{marker}x", f"{marker}y")
        imu_mag = compute_magnitude(imu_data_bandpass_filtered)

        # 4 - Resample to a common freq (VIDEO_FREQ)
        imu_rs = resample_signal(imu_mag, VIDEO_FREQ)

        # 5 - Smooth signals
        window_size = 15
        video_smooth = smooth_signal(video_mag, window=window_size)
        imu_smooth = smooth_signal(imu_rs, window=window_size)

        # 6 - Drop NA
        video_smooth = video_smooth.dropna()
        imu_smooth = imu_smooth.dropna()

        # 7 - Normalize
        NORM_METHOD = "zscore"
        video_norm = normalize_signal(video_smooth, NORM_METHOD)
        imu_norm = normalize_signal(imu_smooth, NORM_METHOD)

        # Uncomment to plot (for debugging)
        plot_debug(video_norm, imu_norm, "Magnitude", labels=["Video", "IMU"])

        # Cross-correlate
        lag_samples, max_corr, similarity_curve = compute_cross_correlation(
            imu_norm, video_norm, 30, LAG_RANGE
        )
        print(f"  Optimal lag (s): {lag_samples/30:.3f}, Correlation: {max_corr:.3f}")

        if SHOW_PLOTS or SAVE_PLOTS:
            plot_similarity_vs_lag(similarity_curve, 30, LAG_RANGE, best_camera, OUTPUT_DIR, save=False)

        # a = teager_kaiser_energy(video_norm)
        # plot_debug(video_norm, a, "Magnitude", labels=["Video", "TKEO"])

        # plot_spectrograms(video_norm, fs=30)

       



##-----------------

#     input_root = r"C:\Users\s4659771\Documents\MyTurn_Project\Data\Processed\P08\Session1_20250304\Video\CT\Camera1\Segments"  
#     segment_folders = [os.path.join(input_root, d) for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
#     for seg_path in segment_folders:
#         print(f"\nProcessing segment folder: {os.path.basename(seg_path)}")
#         # Find CSV file (assume only one .csv per folder)
#         csv_files = [f for f in os.listdir(seg_path) if f.endswith('.csv')]
#         if not csv_files:
#             print("  No CSV file found in this folder, skipping.")
#             continue
#         csv_path = os.path.join(seg_path, csv_files[0])
#         df = pd.read_csv(csv_path)
#         if df.empty:
#             print("  CSV is empty, skipping.")
#             continue
        
#         # Get UNIX start and end from first/last row
#         seg_start = df.iloc[0]['Unix Time']
#         seg_end = df.iloc[-1]['Unix Time']
#         seg_interval = (seg_start - TIME_MARGIN, seg_end + TIME_MARGIN)
        
#         # Determine camera/wrist side from folder or file name
#         camera_label = os.path.basename(os.path.dirname(os.path.dirname(seg_path)))
#         # wrist_side = determine_wrist_side(camera_label)
#         wrist_side = "RH" # this is temporary until I develop the determine_wrist_side method
        
#         # Load IMU and video marker data for interval
#         imu_data = load_imu_data(IMU_DATA_DIR, wrist_side, seg_interval)
#         imu_data['ax'] = -imu_data['ax']
#         video_data = df[['Unix Time', '11_x', '11_y']].copy() #load_video_marker_data(VIDEO_MARKER_DIR, camera_label, seg_interval)
#         video_data['11_y'] = -video_data['11_y'] 

#         # # Compute z-scores on a copy to avoid chained assignment issues
#         # z_scores = zscore(video_data['10_x'].values)

#         # # Parameters
#         # z_thresh = 5.0
#         # window = 5

#         # # Initialize boolean mask (True = keep)
#         # valid_mask = np.full(len(z_scores), False)

#         # # Identify valid windows
#         # for i in range(len(z_scores) - window + 1):
#         #     window_z = z_scores[i:i+window]
#         #     if np.all(np.abs(window_z) < z_thresh):
#         #         valid_mask[i:i+window] = True

#         # # Replace invalid regions with NaN (or optionally 0)
#         # video_data['10_x'] = np.where(valid_mask, video_data['10_x'], np.nan)




#         # # Compute z-scores on a copy to avoid chained assignment issues
#         # z_scores = zscore(video_data['10_y'].values)

#         # # Parameters
#         # z_thresh = 5.0
#         # window = 5

#         # # Initialize boolean mask (True = keep)
#         # valid_mask = np.full(len(z_scores), False)

#         # # Identify valid windows
#         # for i in range(len(z_scores) - window + 1):
#         #     window_z = z_scores[i:i+window]
#         #     if np.all(np.abs(window_z) < z_thresh):
#         #         valid_mask[i:i+window] = True

#         # # Replace invalid regions with NaN (or optionally 0)
#         # video_data['10_y'] = np.where(valid_mask, video_data['10_y'], np.nan)


#         # Plotting
#         plt.figure(figsize=(12, 6))
#         plt.plot(video_data['11_x'], label='ax', alpha=0.8)
#         plt.plot(video_data['11_y'], label='ay', alpha=0.8)
#         # plt.plot(imu_data['az'], label='az', alpha=0.8)
#         # plt.plot(video_mag, label='magnitude', color='black', linewidth=1.5)

#         plt.xlabel("Frame / Sample Index")
#         plt.ylabel("Filtered Acceleration")
#         plt.title("Filtered IMU Axes and Magnitude")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()



#         # High-pass filter each IMU axis
#         imu_data['ax'] = highpass_filter(imu_data['ax'], freq=100, cutoff=0.5)
#         imu_data['ay'] = highpass_filter(imu_data['ay'], freq=100, cutoff=0.5)
#         imu_data['az'] = highpass_filter(imu_data['az'], freq=100, cutoff=0.5)
#         # Low-pass filter each IMU axis
#         imu_data['ax'] = lowpass_filter(imu_data['ax'], freq=100, cutoff=1.0)
#         imu_data['ay'] = lowpass_filter(imu_data['ay'], freq=100, cutoff=1.0)
#         imu_data['az'] = lowpass_filter(imu_data['az'], freq=100, cutoff=1.0)
#         # Compute magnitude
#         imu_mag = compute_magnitude(imu_data)




#         # High-pass filter each video axis
#         video_data['11_x'] = highpass_filter(video_data['11_x'], freq=30, cutoff=0.5)
#         video_data['11_y'] = highpass_filter(video_data['11_y'], freq=30, cutoff=0.5)
#         # Low-pass filter each video axis
#         video_data['11_x'] = lowpass_filter(video_data['11_x'], freq=30, cutoff=1.0)
#         video_data['11_y'] = lowpass_filter(video_data['11_y'], freq=30, cutoff=1.0)
#         # Compute magnitude
#         video_mag = position_to_acceleration(video_data)





#         # # Remove the gravity component from the imu data
#         # imu_mag_hp = highpass_filter(imu_mag, freq=100, cutoff=0.2)
#         # video_mag_hp = highpass_filter(video_mag, freq=30, cutoff=0.2)

#         # imu_mag_lp = lowpass_filter(imu_mag_hp, freq=100, cutoff=1.0)
#         # video_mag_lp = lowpass_filter(video_mag_hp, freq=30, cutoff=1.0)

#         # Resample
#         imu_rs = resample_signal(imu_mag, 30)
#         # video_rs = resample_signal(video_mag, COMMON_FREQ)

#         window_size = 15
#         imu_smooth = smooth_signal(imu_rs, window=window_size)
#         video_smooth = smooth_signal(video_mag, window=window_size)

#         imu_smooth = imu_smooth.dropna()
#         video_smooth = video_smooth.dropna()
        
#         # Normalize
#         NORM_METHOD = "zscore"
#         imu_norm = normalize_signal(imu_smooth, NORM_METHOD)
#         video_norm = normalize_signal(video_smooth, NORM_METHOD)

#         # plot_normalized_signals(imu_norm, video_norm)
#         # plot_normalized_signals(imu_norm, video_norm.assign(Magnitude=-video_norm['Magnitude']))

# #         imu_center = extract_central_window(imu_norm, window_sec=2.0)
# #         video_center = extract_central_window(video_norm, window_sec=2.0)

# #         # Now compute cross-correlation just on these:
# #         lag_samples, max_corr, similarity_curve = compute_cross_correlation(
# #             imu_center, video_center, COMMON_FREQ, LAG_RANGE
# # )
        
#         # Cross-correlate
#         lag_samples, max_corr, similarity_curve = compute_cross_correlation(
#             imu_norm, video_norm, 30, LAG_RANGE
#         )
#         print(f"  Optimal lag (s): {lag_samples/30:.3f}, Correlation: {max_corr:.3f}")
        
#         # Plots
#         if SHOW_PLOTS or SAVE_PLOTS:
#             plot_similarity_vs_lag(similarity_curve, 30, LAG_RANGE, camera_label, OUTPUT_DIR, save=False)
#             plot_aligned_signals(imu_norm, video_norm, lag_samples, COMMON_FREQ, camera_label, OUTPUT_DIR, save=SAVE_PLOTS)

        
#         # # Merge and save CSV
#         # if SAVE_CSV:
#         #     merge_and_save_csv(imu_rs, video_rs, lag_samples, COMMON_FREQ, camera_label, OUTPUT_DIR)

def main():
    """
    Entry point for batch video processing.
    Select the segments to run.
    """
    start_time_all = time.time()

    video_info = get_videos_info(ROOT_DIR, SELECTED_PATIENTS, SELECTED_SESSIONS, SELECTED_SUBFOLDERS)
    
    for video_path, patient, session, affected_side in video_info:
        run_sync(video_path, patient, session, affected_side)

    print("-" * 100)
    print(f"All segments were sync in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time_all))}.")
    print("-" * 100)


if __name__ == "__main__":
    main()
