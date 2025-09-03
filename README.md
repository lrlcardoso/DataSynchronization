# RehabTrack Workflow – Data Synchronization

This is part of the [RehabTrack Workflow](https://github.com/lrlcardoso/RehabTrack_Workflow): a modular Python pipeline for **tracking and analysing physiotherapy movements**, using video and IMU data.  
This module synchronizes and aligns wrist acceleration signals derived from IMU loggers and video-based markers, producing time‑aligned datasets for further analysis.

---

## 📌 Overview

This module performs:
- **Segmentation-based processing** using a user‑defined file of target intervals
- **Signal loading** from IMU CSVs and video‑derived wrist markers
- **Resampling** to match sampling rates
- **Signal normalization** for cross‑source comparison
- **Cross‑correlation** to determine optimal time shift (lag)
- **Alignment** of IMU and video signals
- **Merging** into a single synchronized dataset with both global and segment-relative time
- **Saving** results in CSV format for later stages

**Inputs:**
- IMU logger acceleration data (e.g., WMORE output)
- Video‑derived wrist marker acceleration data
- Segmentation file specifying the intervals to synchronize (same as used in VideoDataProcessing)

**Outputs:**
- Synchronized CSV file containing IMU and video marker signals for each segment
- Figures showing lag correlation and aligned signals

---

## 📂 Repository Structure

```
Data_Synchronization/
├── main.py                   # Main entry point
├── config.py                 # Configurable parameters & paths
├── utils/                    # Helper modules for file I/O, processing, and plotting
│   ├── file_utils.py         # File reading/writing utilities
│   ├── signal_processing.py  # Signal resampling, normalization, and cross-correlation
│   └── plotting.py           # Plot generation for correlation and alignment
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/Data_Synchronization.git
cd Data_Synchronization
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the synchronization pipeline:
```bash
python main.py
```

All main settings (paths, sample rates, etc.) are configured in `config.py`.

**Inputs:**  
- IMU logger CSVs with wrist acceleration data  
- Video marker CSVs with wrist acceleration data  
- Segmentation file listing intervals for synchronization  

**Outputs:**  
- CSV files containing synchronized IMU and video marker data for each segment  
- Plots showing lag vs. correlation and the aligned signals  

---

## 📝 License

Code: [MIT License](LICENSE)  
