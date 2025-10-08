#!/usr/bin/env python3
"""
Simple script to convert dataset to YOLO format.
Usage: python simple_yolo_converter.py
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image


def create_yolo_directories(output_dir):
    """Create YOLO directory structure."""
    dirs = [
        f"{output_dir}/train/images",
        f"{output_dir}/train/labels", 
        f"{output_dir}/val/images",
        f"{output_dir}/val/labels",
        f"{output_dir}/test/images"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Created directories at {output_dir}")


def convert_bbox_to_yolo(x, y, w, h, img_width, img_height):
    """Convert bbox to YOLO format: class x_center y_center width height (normalized)."""
    # Convert to center coordinates
    x_center = x + w / 2
    y_center = y + h / 2
    
    # Normalize
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    
    # Clip to [0, 1]
    x_center_norm = max(0, min(1, x_center_norm))
    y_center_norm = max(0, min(1, y_center_norm))
    width_norm = max(0, min(1, width_norm))
    height_norm = max(0, min(1, height_norm))
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def load_annotations(gt_file):
    """Load ground truth annotations."""
    annotations = {}
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 5:
                frame_id = int(parts[0])
                bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                
                if frame_id not in annotations:
                    annotations[frame_id] = []
                
                annotations[frame_id].append(bbox)
    
    return annotations


def convert_training_data(source_dir, output_dir, train_ratio=0.8):
    """Convert training data and split into train/val."""
    
    # Load annotations
    gt_file = f"{source_dir}/train/gt.txt"
    annotations = load_annotations(gt_file)
    
    print(f"Loaded annotations for {len(annotations)} images")
    
    # Get list of annotated images
    img_dir = f"{source_dir}/train/img"
    annotated_images = []
    
    for img_id in annotations.keys():
        img_file = f"{img_dir}/{img_id:08d}.jpg"
        if os.path.exists(img_file):
            annotated_images.append(img_id)
    
    print(f"Found {len(annotated_images)} valid images")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(annotated_images)
    
    n_train = int(len(annotated_images) * train_ratio)
    train_ids = annotated_images[:n_train]
    val_ids = annotated_images[n_train:]
    
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")
    
    # Process train set
    for img_id in train_ids:
        process_image(img_id, annotations, img_dir, f"{output_dir}/train")
    
    # Process val set  
    for img_id in val_ids:
        process_image(img_id, annotations, img_dir, f"{output_dir}/val")


def process_image(img_id, annotations, img_dir, output_split_dir):
    """Process single image and create YOLO format files."""
    
    # Copy image
    src_img = f"{img_dir}/{img_id:08d}.jpg"
    dst_img = f"{output_split_dir}/images/{img_id:08d}.jpg"
    shutil.copy2(src_img, dst_img)
    
    # Get image dimensions
    with Image.open(src_img) as img:
        img_width, img_height = img.size
    
    # Create label file
    label_file = f"{output_split_dir}/labels/{img_id:08d}.txt"
    
    with open(label_file, 'w') as f:
        for bbox in annotations[img_id]:
            x, y, w, h = bbox
            x_center, y_center, width, height = convert_bbox_to_yolo(
                x, y, w, h, img_width, img_height
            )
            
            # YOLO format: class x_center y_center width height
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def convert_test_data(source_dir, output_dir):
    """Convert test data."""
    
    test_img_dir = f"{source_dir}/test/img"
    output_test_dir = f"{output_dir}/test/images"
    
    # Copy all test images
    for img_file in os.listdir(test_img_dir):
        if img_file.endswith('.jpg'):
            src = f"{test_img_dir}/{img_file}"
            dst = f"{output_test_dir}/{img_file}"
            shutil.copy2(src, dst)
    
    print(f"Copied {len(os.listdir(output_test_dir))} test images")


def create_dataset_yaml(output_dir):
    """Create dataset.yaml for YOLO training."""
    
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['object']
"""
    
    with open(f"{output_dir}/dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml")


def main():
    """Main conversion function."""
    
    source_dir = "/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-1"
    output_dir = "/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-1/yolo_dataset"
    
    print("Converting dataset to YOLO format...")
    
    # Create directories
    create_yolo_directories(output_dir)
    
    # Convert training data
    convert_training_data(source_dir, output_dir)
    
    # Convert test data
    convert_test_data(source_dir, output_dir)
    
    # Create dataset config
    create_dataset_yaml(output_dir)
    
    print("\nConversion completed!")
    print(f"YOLO dataset created at: {output_dir}")
    print("\nTo train with YOLOv8:")
    print("pip install ultralytics")
    print(f"yolo train data={output_dir}/dataset.yaml model=yolov8n.pt epochs=100")


if __name__ == "__main__":
    main()
