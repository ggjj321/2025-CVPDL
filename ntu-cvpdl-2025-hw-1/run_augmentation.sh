#!/bin/bash

# Data Augmentation Script - Rotation
# This script augments the YOLO dataset with rotated images

echo "=============================================="
echo "YOLO Dataset Rotation Augmentation"
echo "=============================================="

SCRIPT_DIR="/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-1"
DATASET_DIR="$SCRIPT_DIR/yolo_dataset"

cd "$SCRIPT_DIR"

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory $DATASET_DIR does not exist!"
    echo "Please run the dataset conversion first."
    exit 1
fi

# Default: augment with 90, 180, 270 degree rotations
echo "Augmenting dataset with rotations: 90°, 180°, 270°"
echo "This will process train and val sets..."
echo ""

# Run augmentation
python augment_rotation.py \
    --dataset "$DATASET_DIR" \
    --angles 90 180 270 \
    --sets train val

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Augmentation completed successfully!"
    echo "=============================================="
    echo ""
    echo "Dataset statistics:"
    echo "  Train images: $(ls $DATASET_DIR/train/images/*.jpg 2>/dev/null | wc -l)"
    echo "  Train labels: $(ls $DATASET_DIR/train/labels/*.txt 2>/dev/null | wc -l)"
    echo "  Val images: $(ls $DATASET_DIR/val/images/*.jpg 2>/dev/null | wc -l)"
    echo "  Val labels: $(ls $DATASET_DIR/val/labels/*.txt 2>/dev/null | wc -l)"
    echo ""
    echo "You can now train YOLO with the augmented dataset!"
else
    echo "ERROR: Augmentation failed!"
    exit 1
fi
