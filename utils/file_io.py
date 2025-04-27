"""
File IO utilities for reading segmentation, IMU, and video marker data.
"""

import os
import pandas as pd

def read_segmentation(segmentation_file):
    """
    Reads the segmentation file and returns a list of segment dicts.
    """
    # TODO: Implement parsing logic
    pass

def load_imu_data(IMU_DATA_DIR, wrist_side, seg_interval):
    """
    Loads IMU data (Unix Time, ax, ay, az) for the given wrist side and segment interval.
    
    Parameters:
        IMU_DATA_DIR (str): Directory containing Logger1.csv and Logger2.csv.
        wrist_side (str): 'RH' for right hand/wrist, 'LH' for left.
        seg_interval (tuple): (start_unix, end_unix) for the segment, in seconds.
    
    Returns:
        pd.DataFrame: Filtered IMU data for this segment.
    """
    # Determine file name
    if wrist_side == "RH":
        file_path = os.path.join(IMU_DATA_DIR, "Logger1.csv")
    elif wrist_side == "LH":
        file_path = os.path.join(IMU_DATA_DIR, "Logger2.csv")
    else:
        raise ValueError(f"Unknown wrist_side: {wrist_side}")
    
    # Read the IMU CSV
    df = pd.read_csv(file_path)
    
    # Ensure the column names are as expected
    required_cols = ['Unix Time', 'ax', 'ay', 'az']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {file_path}")
    
    # Filter by segment interval (as close as possible)
    start, end = seg_interval
    df_filtered = df[(df['Unix Time'] >= start) & (df['Unix Time'] <= end)].copy()
    
    # Select relevant columns
    imu_data = df_filtered[required_cols].reset_index(drop=True)
    return imu_data

def load_video_marker_data(marker_dir, camera, interval):
    """
    Loads video marker acceleration magnitude for the specified camera and interval.
    """
    # TODO: Implement loading and interval extraction
    pass

def determine_wrist_side(camera_label):
    """
    Determines if the camera corresponds to left or right wrist.
    """
    # TODO: Implement logic based on camera_label naming convention
    pass
