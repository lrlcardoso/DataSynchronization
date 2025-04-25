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
from utils.file_io import load_imu_data, load_video_marker_data, determine_wrist_side
from utils.signal_processing import resample_signal, normalize_signal, compute_cross_correlation
from utils.plotting import plot_similarity_vs_lag, plot_aligned_signals
from utils.data_merge import merge_and_save_csv

def main():
    input_root = r"C:\Users\s4659771\Documents\MyTurn_Project\Data\Processed\P01\Session2_20250210\Video\VR\Camera1\Segments"  # Adjust as needed or move to config.py
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

        print(seg_interval)
        
        # Determine camera/wrist side from folder or file name
        camera_label = os.path.basename(os.path.dirname(os.path.dirname(seg_path)))
        print(camera_label)
        # wrist_side = determine_wrist_side(camera_label)
        
        # # Load IMU and video marker data for interval
        # imu_data = load_imu_data(IMU_DATA_DIR, wrist_side, seg_interval)
        # video_data = load_video_marker_data(VIDEO_MARKER_DIR, camera_label, seg_interval)
        
        # # Resample
        # imu_rs = resample_signal(imu_data, IMU_FREQ, COMMON_FREQ)
        # video_rs = resample_signal(video_data, VIDEO_FREQ, COMMON_FREQ)
        
        # # Normalize
        # imu_norm = normalize_signal(imu_rs, NORM_METHOD)
        # video_norm = normalize_signal(video_rs, NORM_METHOD)
        
        # # Cross-correlate
        # lag_samples, max_corr, similarity_curve = compute_cross_correlation(
        #     imu_norm, video_norm, COMMON_FREQ, LAG_RANGE
        # )
        # print(f"  Optimal lag (s): {lag_samples/COMMON_FREQ:.3f}, Correlation: {max_corr:.3f}")
        
        # # Plots
        # if SHOW_PLOTS or SAVE_PLOTS:
        #     plot_similarity_vs_lag(similarity_curve, COMMON_FREQ, LAG_RANGE, camera_label, OUTPUT_DIR, save=SAVE_PLOTS)
        #     plot_aligned_signals(imu_norm, video_norm, lag_samples, COMMON_FREQ, camera_label, OUTPUT_DIR, save=SAVE_PLOTS)
        
        # # Merge and save CSV
        # if SAVE_CSV:
        #     merge_and_save_csv(imu_rs, video_rs, lag_samples, COMMON_FREQ, camera_label, OUTPUT_DIR)

if __name__ == "__main__":
    main()
