# Colab Setup: WLASL100 Data Sanity Check

## 1) Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2) Clone repo (optional if already present)
```bash
!git clone https://github.com/Neolemoz/wlasl_demo.git
%cd wlasl_demo
```

## 3) Install minimal dependencies
```bash
!pip install -q opencv-python numpy pandas matplotlib
```

## 4) Set parameters in `colab/wlasl100_data_sanity.py`
Edit the constants at the top of the file:
```python
DATA_ROOT = "/content/drive/MyDrive/<your_dataset_root>"
ANN_PATH = "/content/drive/MyDrive/<your_dataset_root>/<annotations_file>.json"
LABELS_PATH = "/content/drive/MyDrive/<your_export_or_weights>/labels.json"
SPLIT = "train"  # or "val"
SAMPLE_VIS_N = 5
DECODE_CHECK_N = 50
SEED = 1337
```

## 5) Run
```bash
!python colab/wlasl100_data_sanity.py
```

## Notes
- `LABELS_PATH` should be the same label mapping used for training/export.
- PR#4 expects JSON annotations in the repo-style format.
- CSV annotations are not supported by default (`ALLOW_CSV=False`).
- If you intentionally enable CSV later, set `ALLOW_CSV=True` and update parsing consciously.
- If `ffmpeg/ffprobe` are not installed, this script still runs because decode uses OpenCV.

## DONE checklist
- [x] prints min/median/max clips per class
- [x] prints 10 smallest / 10 largest classes
- [x] shows 5 sample videos (frames)
- [x] prints missing_file_count + decode_failed_count
- [x] confirms label<->id mapping used for training/export
