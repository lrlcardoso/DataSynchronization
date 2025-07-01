"""
==============================================================================
Title:          Signal Synchronization Config
Description:    Stores global configuration variables for the signal-video 
                synchronization pipeline, including patient/session filters, 
                signal filtering parameters, and plotting/saving options.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-25
Version:        1.1
==============================================================================
Usage:
    This file is intended to be imported into synchronization-related scripts 
    (e.g., main_sync.py, sync_utils.py) to access global configuration options.

Dependencies:
    - Python >= 3.x

Notes:
    - Edit SELECTED_PATIENTS, SESSIONS, or SEGMENTS to restrict processing scope.
    - Use SHOW_PLOTS and SAVE_PLOTS to control debug visualization behavior.
    - If using an external venv. that runs for the entire project, run:
      & 'C:\\Users\\s4659771\\Documents\\MyTurn_Project\\.venv\\Scripts\\Activate.ps1'

Changelog:
    - v1.0: [2025-04-25] Initial release
    - v1.1: [2025-06-27] Organized necessary variables
==============================================================================
"""

ROOT_DIR = r"C:\Users\s4659771\Documents\MyTurn_Project\Data"
SELECTED_PATIENTS = ["P02"]
SELECTED_SESSIONS = ["Session1"]
SELECTED_SUBFOLDERS = ["FMA_and_VR", "VR", "CT"]
SELECTED_SEGMENTS = [15]

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