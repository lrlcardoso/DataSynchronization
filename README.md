# Signal Synchronization and Alignment Pipeline

This project synchronizes and aligns wrist acceleration signals from two different sources:

- **IMU logger (WMORE):** High-frequency accelerometer data (e.g., 100 Hz).
- **Video-derived marker:** Wrist marker acceleration extracted from video frames (e.g., 30 Hz).

The pipeline identifies the optimal time shift to align both signals using cross-correlation, then merges and saves the synchronized data for further analysis.

---

## Project Structure

your_project/
│
├── main.py
├── config.py
├── README.md
└── Utils/
    ├── __init__.py
    ├── file_io.py
    ├── signal_processing.py
    ├── plotting.py
    └── data_merge.py

---

## Usage

```bash
python main.py
```

All main settings (paths, sample rates, etc.) can be configured in `config.py`.

---

## Pipeline Steps

1. Read the segmentation file to extract all intervals.
2. Load IMU and video marker data for each segment, with ±1 second margin.
3. Identify the correct wrist logger (left/right).
4. Resample both signals to the same frequency.
5. Normalize both signals.
6. Compute cross-correlation and find the optimal lag.
7. Plot similarity vs. lag and the aligned signals.
8. Save a CSV with synchronized IMU and video marker data, including global UNIX time and segment-relative time.

---

## Requirements

- Python >= 3.7
- numpy
- pandas
- scipy
- matplotlib
- natsort
- openpyxl

Install requirements (if needed):

```bash
python -m pip install -r requirements.txt
```

---

## Author

Lucas R. L. Cardoso
