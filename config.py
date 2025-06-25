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

ROOT_DIR = r"C:\Users\s4659771\Documents\MyTurn_Project\Data"
SELECTED_PATIENTS = ["P05"]
SELECTED_SESSIONS = ["Session1"]
SELECTED_SUBFOLDERS = ["FMA_and_VR", "VR", "CT"]
SELECTED_SEGMENTS = None

# --- File Paths ---
SEGMENTATION_FILE = "data/segments.txt"
IMU_DATA_DIR = r"C:\Users\s4659771\Documents\MyTurn_Project\Data\Processed\P08\Session1_20250304\WMORE"
VIDEO_MARKER_DIR = r"C:\Users\s4659771\Documents\MyTurn_Project\Data\Processed\P08\Session1_20250304\Video\CT"
OUTPUT_DIR = "results/"

# --- Processing Parameters ---
IMU_FREQ = 100           # Hz
VIDEO_FREQ = 30          # Hz
TIME_MARGIN = 0.0        # seconds before/after each segment
FILTER_LOW_CUT = 0.5
FILTER_HIGH_CUT = 1.0
FILTER_ORDER = 2

# --- Normalization ---
NORM_METHOD = "zscore"   # 'zscore' or 'minmax'

# --- Correlation ---
LAG_RANGE = 5          # seconds, search +/- this window

# --- Plotting ---
PLOT_FORMAT = "png"
SHOW_PLOTS = True
SAVE_PLOTS = False

# --- CSV Output ---
SAVE_CSV = True
