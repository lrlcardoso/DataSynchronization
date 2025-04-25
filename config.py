"""
==============================================================================
Title:          Signal Synchronization Config
Description:    Stores global configuration for synchronization pipeline
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-25
Version:        1.0
==============================================================================
"""

# --- File Paths ---
SEGMENTATION_FILE = "data/segments.txt"
IMU_DATA_DIR = "data/imu/"
VIDEO_MARKER_DIR = "data/video_markers/"
OUTPUT_DIR = "results/"

# --- Processing Parameters ---
IMU_FREQ = 100           # Hz
VIDEO_FREQ = 30          # Hz
COMMON_FREQ = 100        # Hz (target frequency for resampling)
TIME_MARGIN = 1.0        # seconds before/after each segment

# --- Normalization ---
NORM_METHOD = "zscore"   # 'zscore' or 'minmax'

# --- Correlation ---
LAG_RANGE = 1.5          # seconds, search +/- this window

# --- Plotting ---
PLOT_FORMAT = "png"
SHOW_PLOTS = True
SAVE_PLOTS = True

# --- CSV Output ---
SAVE_CSV = True
