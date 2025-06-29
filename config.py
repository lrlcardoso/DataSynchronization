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
SELECTED_PATIENTS = ["P06"]
SELECTED_SESSIONS = ["Session2"]
SELECTED_SUBFOLDERS = ["FMA_and_VR", "VR", "CT"]
SELECTED_SEGMENTS = None

IMU_FREQ = 100               # Hz
VIDEO_FREQ = 30              # Hz
FILTER_LOW_CUT = 0.5         # Hz
FILTER_HIGH_CUT = 1.0        # Hz
FILTER_ORDER = 2
LAG_RANGE = 5                # seconds, search +/- this window
WINDOW_SIZE = 3*VIDEO_FREQ   # seconds of window size for final smooth

# --- Plotting ---
SHOW_PLOTS = False
SHOW_DEBG_PLOTS = False

# --- Saving ---
OUTPUT_DIR = "ReadyToAnalyse"
SAVE_PLOTS = True
SAVE_CSV = True