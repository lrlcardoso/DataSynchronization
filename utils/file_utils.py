"""
==============================================================================
Title:          File IO Utilities
Description:    Provides helper functions for retrieving video session info 
                and loading IMU data for a given segment based on the affected side.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-25
Version:        1.0
==============================================================================
Usage:
    from utils.file_utils import get_videos_info, load_imu_data

Dependencies:
    - Python >= 3.x
    - Required libraries: os, pandas

Notes:
    - Assumes patient side information is stored in a tab-delimited file:
        ROOT_DIR/Processed/side_info.txt
    - IMU logger files (Logger1.csv, Logger2.csv) must exist in the WMORE folder 
      one level above the session's video folder.

Changelog:
    - v1.0: [2025-04-25] Initial release
==============================================================================
"""

import os
import pandas as pd

def get_videos_info(root_dir, patients, sessions, subfolders):
    """
    Finds all subfolder video directories for selected patients and sessions,
    and associates them with the patient's affected side.

    Parameters:
    - root_dir: str, base path to the dataset.
    - patients: list of str, patient IDs (e.g., 'P01').
    - sessions: list of str, session prefixes (e.g., 'Session1').
    - subfolders: list of str, subdirectories under 'Video' to include (e.g., 'CT', 'VR').

    Returns:
    - entries: list of tuples (video_path, patient, session_dir, affected_side)
    """
    side_file_path = os.path.join(root_dir, "Processed", "side_info.txt")

    if not os.path.isfile(side_file_path):
        raise FileNotFoundError(f"Affected side file not found at: {side_file_path}")

    # Load patient side mapping from tab-separated text file
    try:
        sides_df = pd.read_csv(side_file_path, sep='\t', header=None, names=["patient", "side"], dtype=str)
    except Exception as e:
        raise ValueError(f"Failed to load side info file: {e}")

    # Strip whitespace to prevent accidental formatting issues
    sides_df["patient"] = sides_df["patient"].str.strip()
    sides_df["side"] = sides_df["side"].str.strip()

    # Convert to dictionary for fast lookup
    patient_side_map = dict(zip(sides_df.patient, sides_df.side))

    entries = []

    for patient in patients:
        if patient not in patient_side_map:
            raise ValueError(f"Affected side not defined for patient: {patient}")

        affected_side = patient_side_map[patient]
        patient_path = os.path.join(root_dir, "Processed", patient)
        if not os.path.isdir(patient_path):
            raise FileNotFoundError(f"Patient folder not found: {patient_path}")

        for session in sessions:
            session_dirs = [d for d in os.listdir(patient_path) if d.startswith(f"{session}_")]
            if not session_dirs:
                raise FileNotFoundError(f"No session matching '{session}_*' in: {patient_path}")

            for session_dir in session_dirs:
                session_path = os.path.join(patient_path, session_dir, "Video")
                if not os.path.isdir(session_path):
                    raise FileNotFoundError(f"Video folder not found: {patient} / {session_dir} / Video")

                for subdir in os.listdir(session_path):
                    if subdir in subfolders:
                        subdir_path = os.path.join(session_path, subdir)
                        if os.path.isdir(subdir_path):
                            entries.append((subdir_path, patient, session_dir, affected_side))

    return entries

def load_imu_data(IMU_DATA_DIR, affected_side, seg_interval):
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
    if affected_side == "R" or affected_side == "B":
        file_path = os.path.join(IMU_DATA_DIR, "Logger1.csv")
    elif affected_side == "L":
        file_path = os.path.join(IMU_DATA_DIR, "Logger2.csv")
    else:
        raise ValueError(f"Unknown affected_side: {affected_side}")
    
    # Read the IMU CSV
    df = pd.read_csv(file_path)
    
    # Ensure the column names are as expected
    required_cols = ['Unix Time', 'ax', 'ay', 'az']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {file_path}")
    
    # Filter by segment interval (as close as possible)
    start, end = seg_interval
    print(seg_interval)
    df_filtered = df[(df['Unix Time'] >= start) & (df['Unix Time'] <= end)].copy()
    
    # Select relevant columns
    imu_data = df_filtered[required_cols].reset_index(drop=True)
    
    return imu_data