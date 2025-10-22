#!/bin/bash

# YOLO Dataset Conversion and Training Script
# This script will convert your dataset to YOLO format and train a model

echo "==============================================="
echo "YOLO Object Detection Pipeline"
echo "==============================================="

# Set paths
SCRIPT_DIR="/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-1"
YOLO_DATASET_DIR="$SCRIPT_DIR/yolo_dataset"

echo "Working directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

# Step 1: (Optional) Install dependencies
# echo "Step 1: Installing dependencies..."
# pip install ultralytics pandas pillow pyyaml

# Step 2: (Optional) Convert dataset to YOLO format
# echo "Step 2: Converting dataset to YOLO format..."
# python simple_yolo_converter.py
# if [ ! -f "$YOLO_DATASET_DIR/dataset.yaml" ]; then
#     echo "ERROR: Dataset conversion failed!"
#     exit 1
# fi
# echo "Dataset conversion completed!"

# Step 3: (Optional) Train YOLO model
# EPOCHS=100
# IMG_SIZE=640
# BATCH_SIZE=8
# yolo detect train model=yolo12s.yaml data="$YOLO_DATASET_DIR/dataset.yaml" epochs=$EPOCHS imgsz=$IMG_SIZE batch=$BATCH_SIZE pretrained=False

# # yolo detect train model=yolo11n.yaml data=/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-1/yolo_dataset/dataset.yaml epochs=200 imgsz=640 batch=8 pretrained=False

# Identify the latest training run (ultralytics auto-increments train, train2, ...)
LAST_RUN_DIR=$(ls -dt "$SCRIPT_DIR"/runs/detect/train* 2>/dev/null | head -1)
if [ -z "$LAST_RUN_DIR" ]; then
    echo "ERROR: No training run directory found!"
    exit 1
fi

echo "Latest run directory: $LAST_RUN_DIR"
RUNS_DIR="$LAST_RUN_DIR/weights"

# Check if training completed and best model exists
if [ ! -f "$RUNS_DIR/best.pt" ]; then
    echo "ERROR: best.pt not found in $RUNS_DIR"
    exit 1
fi

echo "Found model: $RUNS_DIR/best.pt"

# Step 4: Generate predictions (submission needed)
echo "Generating predictions on test set..."
PRED_CONF=0.25
yolo predict model="$RUNS_DIR/best.pt" source="$SCRIPT_DIR/test/img" save_txt=True save_conf=True conf=$PRED_CONF \
    project="$SCRIPT_DIR/runs/detect" name="predict" exist_ok=True > /dev/null 2>&1 || {
    echo "ERROR: Prediction step failed"; exit 1; }

echo "Converting predictions to submission CSV..."
python - <<'PYCODE'
import os
from pathlib import Path
import pandas as pd
from PIL import Image

SCRIPT_DIR = Path("/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-1")
TEST_IMG_DIR = SCRIPT_DIR / "test" / "img"
PRED_ROOT = SCRIPT_DIR / "runs" / "detect"
TARGET_PRED_DIR = PRED_ROOT / "predict"

# If predict2, predict3 etc. also exist, include them ordered by mtime
predict_dirs = []
if PRED_ROOT.exists():
    predict_dirs = [d for d in PRED_ROOT.glob("predict*") if d.is_dir()]
    predict_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

if not predict_dirs:
    print("WARNING: No predict* directory found under runs/detect. Creating empty submission.")
    predict_dirs = [TARGET_PRED_DIR]

PRED_DIR = predict_dirs[0]
LABEL_DIR = PRED_DIR / "labels"
print(f"Using prediction directory: {PRED_DIR}")
if not LABEL_DIR.exists():
    print(f"WARNING: labels directory not found: {LABEL_DIR}. Will create empty submission.")

rows = []
for img_file in sorted(TEST_IMG_DIR.glob('*.jpg')):
    try:
        img_id = int(img_file.stem)
    except ValueError:
        continue
    label_file = LABEL_DIR / f"{img_file.stem}.txt"
    preds = []
    if label_file.exists():
        with Image.open(img_file) as im:
            w, h = im.size
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cls, xc, yc, bw, bh, conf = parts[:6]
                try:
                    xc = float(xc) * w
                    yc = float(yc) * h
                    bw = float(bw) * w
                    bh = float(bh) * h
                    x = xc - bw / 2
                    y = yc - bh / 2
                except ValueError:
                    continue
                preds.append(f"{conf} {x:.2f} {y:.2f} {bw:.2f} {bh:.2f} 0")
    rows.append({"Image_ID": img_id, "PredictionString": " ".join(preds)})

if len(rows) == 0:
    print("ERROR: No test images found!")
    raise SystemExit(1)

df = pd.DataFrame(rows).sort_values("Image_ID")
out_csv = SCRIPT_DIR / "yolo_submission.csv"
df.to_csv(out_csv, index=False)
print(f"Saved submission: {out_csv}")
print(f"Total images: {len(df)}")
print(f"Images with detections: {(df['PredictionString']!='').sum()}")
PYCODE

STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to create submission CSV"
    exit 1
fi

echo "Submission file ready: $SCRIPT_DIR/yolo_submission.csv"
echo "Done."
