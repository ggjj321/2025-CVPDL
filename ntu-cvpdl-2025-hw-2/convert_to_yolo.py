#!/usr/bin/env python3
"""
Convert CVPDL_hw2 dataset to YOLO format
Input format: <class>,<x>,<y>,<width>,<height> (absolute coordinates)
Output format: <class> <x_center> <y_center> <width> <height> (normalized 0-1)
"""

import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box from absolute coordinates to YOLO format
    Input: [class, x, y, w, h] (absolute pixels, top-left corner)
    Output: [class, x_center, y_center, w, h] (normalized 0-1)
    """
    class_id, x, y, w, h = bbox
    
    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # Normalize width and height
    width_norm = w / img_width
    height_norm = h / img_height
    
    # Clip to [0, 1] range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width_norm = max(0, min(1, width_norm))
    height_norm = max(0, min(1, height_norm))
    
    return class_id, x_center, y_center, width_norm, height_norm


def parse_annotation_file(ann_file):
    """Parse annotation file and return list of bounding boxes"""
    bboxes = []
    with open(ann_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                class_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                bboxes.append([class_id, x, y, w, h])
    return bboxes


def detect_classes(source_dir):
    """Detect all unique classes in the dataset"""
    classes = set()
    txt_files = list(Path(source_dir).glob('*.txt'))
    
    for txt_file in txt_files:
        bboxes = parse_annotation_file(txt_file)
        for bbox in bboxes:
            classes.add(bbox[0])
    
    return sorted(list(classes))


def convert_dataset(source_dir, output_dir, split='train', val_split=0.1):
    """
    Convert dataset to YOLO format
    
    Args:
        source_dir: Path to source directory containing images and txt files
        output_dir: Path to output YOLO dataset directory
        split: 'train' or 'test'
        val_split: Percentage of training data to use for validation
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Get all image files
    image_files = sorted(list(source_dir.glob('*.png')) + list(source_dir.glob('*.jpg')))
    
    print(f"\nProcessing {split} set: {len(image_files)} images")
    
    if split == 'train':
        # Split into train and val
        val_count = int(len(image_files) * val_split)
        train_files = image_files[val_count:]
        val_files = image_files[:val_count]
        
        print(f"  Train: {len(train_files)} images")
        print(f"  Val: {len(val_files)} images")
        
        # Process train set
        _convert_split(train_files, source_dir, output_dir / 'train')
        
        # Process val set
        _convert_split(val_files, source_dir, output_dir / 'val')
        
    else:  # test
        # Test set (no annotations)
        _convert_split(image_files, source_dir, output_dir / 'test', has_labels=False)


def _convert_split(image_files, source_dir, output_split_dir, has_labels=True):
    """Convert a split of the dataset"""
    
    # Create output directories
    img_out_dir = output_split_dir / 'images'
    label_out_dir = output_split_dir / 'labels'
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    if has_labels:
        label_out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(image_files, desc=f"Converting {output_split_dir.name}"):
        # Copy image
        img_out_path = img_out_dir / img_path.name
        shutil.copy(img_path, img_out_path)
        
        if has_labels:
            # Read image dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # Read annotations
            ann_file = source_dir / (img_path.stem + '.txt')
            
            if ann_file.exists():
                bboxes = parse_annotation_file(ann_file)
                
                # Convert to YOLO format
                yolo_bboxes = []
                for bbox in bboxes:
                    yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                    yolo_bboxes.append(yolo_bbox)
                
                # Write YOLO format labels
                label_out_path = label_out_dir / (img_path.stem + '.txt')
                with open(label_out_path, 'w') as f:
                    for class_id, x_c, y_c, w, h in yolo_bboxes:
                        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            else:
                # Create empty label file if annotation doesn't exist
                label_out_path = label_out_dir / (img_path.stem + '.txt')
                label_out_path.touch()


def create_yaml_file(output_dir, class_list):
    """Create dataset.yaml file for YOLO"""
    output_dir = Path(output_dir)
    
    # Create class names based on detected classes
    num_classes = len(class_list)
    class_names = [f'class_{i}' for i in class_list]
    
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': class_names
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\nCreated dataset.yaml at {yaml_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Class IDs: {class_list}")
    print(f"Class names: {class_names}")
    return yaml_path


def main():
    # Paths
    base_dir = Path('/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-2')
    source_train_dir = base_dir / 'CVPDL_hw2' / 'CVPDL_hw2' / 'train'
    source_test_dir = base_dir / 'CVPDL_hw2' / 'CVPDL_hw2' / 'test'
    output_dir = base_dir / 'yolo_dataset'
    
    print("="*60)
    print("CVPDL_hw2 to YOLO Format Converter")
    print("="*60)
    print(f"Source train dir: {source_train_dir}")
    print(f"Source test dir: {source_test_dir}")
    print(f"Output dir: {output_dir}")
    print("="*60)
    
    # Check if source directories exist
    if not source_train_dir.exists():
        print(f"Error: Train directory not found: {source_train_dir}")
        return
    
    if not source_test_dir.exists():
        print(f"Error: Test directory not found: {source_test_dir}")
        return
    
    # Detect classes from training data
    print("\nDetecting classes from training data...")
    class_list = detect_classes(source_train_dir)
    print(f"Found {len(class_list)} classes: {class_list}")
    
    # Convert train set (will be split into train/val)
    convert_dataset(source_train_dir, output_dir, split='train', val_split=0.1)
    
    # Convert test set
    convert_dataset(source_test_dir, output_dir, split='test')
    
    # Create dataset.yaml
    create_yaml_file(output_dir, class_list)
    
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/")
    print(f"    │   └── labels/")
    print(f"    ├── val/")
    print(f"    │   ├── images/")
    print(f"    │   └── labels/")
    print(f"    ├── test/")
    print(f"    │   └── images/")
    print(f"    └── dataset.yaml")
    
    # Print statistics
    train_imgs = len(list((output_dir / 'train' / 'images').glob('*')))
    val_imgs = len(list((output_dir / 'val' / 'images').glob('*')))
    test_imgs = len(list((output_dir / 'test' / 'images').glob('*')))
    
    print(f"\nDataset statistics:")
    print(f"  Train images: {train_imgs}")
    print(f"  Val images: {val_imgs}")
    print(f"  Test images: {test_imgs}")
    print(f"  Total: {train_imgs + val_imgs + test_imgs}")
    print(f"  Classes: {len(class_list)} ({class_list})")


if __name__ == "__main__":
    main()
