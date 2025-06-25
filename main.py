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
from natsort import natsorted

# === Project Modules ===
from config import (
    TIME_MARGIN, LAG_RANGE, IMU_FREQ, VIDEO_FREQ,
    FILTER_LOW_CUT, FILTER_HIGH_CUT, FILTER_ORDER,
    ROOT_DIR, SELECTED_PATIENTS, SELECTED_SESSIONS,
    SELECTED_SUBFOLDERS, SELECTED_SEGMENTS, SHOW_PLOTS,
    SAVE_PLOTS, OUTPUT_DIR
)
from utils.file_utils import load_imu_data, get_videos_info
from utils.signal_processing import (
    teager_kaiser_energy, smooth_signal,
    lowpass_filter_avoiding_gaps, resample_signal,
    compute_magnitude, highpass_filter, normalize_signal,
    lowpass_filter, position_to_acceleration,
    compute_cross_correlation
)
from utils.plotting import (
    plot_debug, plot_similarity_vs_lag,
    plot_aligned_signals, plot_spectrograms
)
from utils.data_merge import merge_and_save_csv

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
        # plot_debug(video_data_resampled, video_data_lowpass_filtered, markers=(f"{marker}y",f"{marker}y"), labels=["Video - resampled", "Video - filtered"])

        # Apply a highpass filter to match to the IMU that must be highpassed filtered to get rid of the gravity effect
        video_data_bandpass_filtered = highpass_filter(video_data_lowpass_filtered, fs=VIDEO_FREQ, cutoff=FILTER_LOW_CUT, order=FILTER_ORDER)

        # Resample to IMU data to make sure there is no gap
        imu_data_resampled = resample_signal(imu_data, IMU_FREQ, fill_missing_with_zero=False)

        # Apply a lowpass filter to reduce noise, matching the video_data
        imu_data_lowpass_filtered = lowpass_filter(imu_data_resampled, fs=IMU_FREQ, cutoff=FILTER_HIGH_CUT, order=FILTER_ORDER)

        # Uncomment to plot (for debugging)
        # plot_debug(imu_data_resampled, imu_data_lowpass_filtered, markers=("ax","ax"), labels=["IMU - resampled", "IMU - filtered"])

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
        # plot_debug(video_norm, imu_norm,  markers=("Magnitude","Magnitude") , labels=["Video", "IMU"])
        plot_debug(video_norm, video_data_resampled, markers=("Magnitude", f"{marker}y"), labels=["Video", "resampled"])

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
        # plot_aligned_signals(imu_norm, video_norm, lag_samples, COMMON_FREQ, camera_label, OUTPUT_DIR, save=SAVE_PLOTS)

        # # Merge and save CSV
        # if SAVE_CSV:
        #   merge_and_save_csv(imu_rs, video_rs, lag_samples, COMMON_FREQ, camera_label, OUTPUT_DIR)

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
