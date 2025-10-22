#!/bin/bash

# Complete YOLO Pipeline for CVPDL_hw2
# This script converts data, trains YOLO, and generates submission

echo "=============================================="
echo "CVPDL_hw2 YOLO Pipeline"
echo "=============================================="

BASE_DIR="/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-2"
YOLO_DATASET_DIR="$BASE_DIR/yolo_dataset"

cd "$BASE_DIR"

# Step 1: Convert dataset to YOLO format
# echo ""
# echo "Step 1: Converting dataset to YOLO format..."
# python convert_to_yolo.py

# if [ ! -f "$YOLO_DATASET_DIR/dataset.yaml" ]; then
#     echo "ERROR: Dataset conversion failed!"
#     exit 1
# fi

# echo "Dataset conversion completed!"

# Step 2: Train YOLO model
echo ""
echo "Step 2: Training YOLO model..."
echo "This may take a while depending on your hardware..."

# Training parameters
EPOCHS=100
IMG_SIZE=1080
BATCH_SIZE=16
MODEL="yolo12n.yaml"  # nano model

echo "Training parameters:"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Image size: $IMG_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Train YOLO
yolo detect train \
    model=$MODEL \
    data="$YOLO_DATASET_DIR/dataset.yaml" \
    epochs=$EPOCHS \
    imgsz=$IMG_SIZE \
    batch=$BATCH_SIZE \
    device=0 \
    patience=20 \
    save=True \
    plots=True \
    pretrained=False

# Find latest training run
LAST_TRAIN_DIR=$(ls -dt "$BASE_DIR"/runs/detect/train* 2>/dev/null | head -1)

if [ -z "$LAST_TRAIN_DIR" ] || [ ! -f "$LAST_TRAIN_DIR/weights/best.pt" ]; then
    echo "ERROR: Training failed or model not found!"
    exit 1
fi

echo ""
echo "Training completed!"
echo "Best model: $LAST_TRAIN_DIR/weights/best.pt"

# Step 3: Run prediction on test set
echo ""
echo "Step 3: Running predictions on test set..."

TEST_IMG_DIR="$BASE_DIR/CVPDL_hw2/CVPDL_hw2/test"

yolo predict \
    model="$LAST_TRAIN_DIR/weights/best.pt" \
    source="$TEST_IMG_DIR" \
    save_txt=True \
    save_conf=True \
    conf=0.25 \
    iou=0.45 \
    project="$BASE_DIR/runs/detect" \
    name="predict" \
    exist_ok=True

# Check if prediction was successful
LAST_PRED_DIR=$(ls -dt "$BASE_DIR"/runs/detect/predict* 2>/dev/null | head -1)

if [ -z "$LAST_PRED_DIR" ] || [ ! -d "$LAST_PRED_DIR/labels" ]; then
    echo "ERROR: Prediction failed!"
    exit 1
fi

echo "Prediction completed!"
echo "Predictions saved to: $LAST_PRED_DIR"

# Step 4: Generate submission file
echo ""
echo "Step 4: Generating submission CSV..."

python generate_submission.py

if [ ! -f "$BASE_DIR/submission.csv" ]; then
    echo "ERROR: Submission generation failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo ""
echo "Results:"
echo "  Trained model: $LAST_TRAIN_DIR/weights/best.pt"
echo "  Predictions: $LAST_PRED_DIR"
echo "  Submission file: $BASE_DIR/submission.csv"
echo ""
echo "Next steps:"
echo "  1. Check submission.csv format"
echo "  2. Submit to competition platform"
echo "  3. To retrain with different parameters:"
echo "     yolo detect train model=yolo11n.pt data=$YOLO_DATASET_DIR/dataset.yaml epochs=200"
echo ""
