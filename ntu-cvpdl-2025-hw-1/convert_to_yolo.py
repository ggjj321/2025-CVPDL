#!/usr/bin/env python3
"""
Script to convert training and test data to YOLO format.

YOLO format requirements:
- Images and labels in separate directories
- Label files have same name as image files but with .txt extension
- Each line in label file: <class> <x_center> <y_center> <width> <height>
- All coordinates are normalized (0-1)
- Directory structure:
  datasets/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/
  │   ├── images/
  │   └── labels/
  └── test/
      └── images/
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
import random


class YOLOConverter:
    def __init__(self, source_dir, output_dir, train_split=0.8, val_split=0.2):
        """
        Args:
            source_dir: Path to the original dataset
            output_dir: Path to output YOLO format dataset
            train_split: Proportion for training set
            val_split: Proportion for validation set
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.val_split = val_split
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create YOLO format directory structure."""
        dirs = [
            self.output_dir / 'train' / 'images',
            self.output_dir / 'train' / 'labels',
            self.output_dir / 'val' / 'images', 
            self.output_dir / 'val' / 'labels',
            self.output_dir / 'test' / 'images'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Created YOLO directory structure at {self.output_dir}")
    
    def parse_gt_file(self, gt_file_path):
        """
        Parse ground truth file and return annotations dictionary.
        Format: <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>
        """
        annotations = {}
        
        with open(gt_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 5:
                    continue
                
                try:
                    frame_id = int(parts[0])
                    bb_left = float(parts[1])
                    bb_top = float(parts[2])
                    bb_width = float(parts[3])
                    bb_height = float(parts[4])
                    
                    if frame_id not in annotations:
                        annotations[frame_id] = []
                    
                    annotations[frame_id].append({
                        'bbox': [bb_left, bb_top, bb_width, bb_height],
                        'class': 0  # Single class detection
                    })
                    
                except (ValueError, IndexError):
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
        
        return annotations
    
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """
        Convert bbox from [x, y, w, h] to YOLO format [x_center, y_center, width, height].
        All values normalized to 0-1.
        """
        x, y, w, h = bbox
        
        # Convert to center coordinates
        x_center = x + w / 2
        y_center = y + h / 2
        
        # Normalize coordinates
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = w / img_width
        height_norm = h / img_height
        
        # Clip to valid range [0, 1]
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))
        
        return [x_center_norm, y_center_norm, width_norm, height_norm]
    
    def convert_training_data(self):
        """Convert training data to YOLO format."""
        
        # Parse ground truth
        gt_file = self.source_dir / 'train' / 'gt.txt'
        annotations = self.parse_gt_file(gt_file)
        
        print(f"Loaded annotations for {len(annotations)} images")
        
        # Get all training images
        train_img_dir = self.source_dir / 'train' / 'img'
        image_files = list(train_img_dir.glob('*.jpg'))
        
        # Filter images that have annotations
        annotated_images = []
        for img_file in image_files:
            img_id = int(img_file.stem)
            if img_id in annotations:
                annotated_images.append((img_file, img_id))
        
        print(f"Found {len(annotated_images)} annotated images")
        
        # Shuffle and split data
        random.seed(42)
        random.shuffle(annotated_images)
        
        n_train = int(len(annotated_images) * self.train_split)
        train_data = annotated_images[:n_train]
        val_data = annotated_images[n_train:]
        
        print(f"Split: {len(train_data)} train, {len(val_data)} val")
        
        # Convert training set
        self._convert_split(train_data, annotations, 'train')
        
        # Convert validation set
        self._convert_split(val_data, annotations, 'val')
    
    def _convert_split(self, data, annotations, split_name):
        """Convert a data split to YOLO format."""
        
        print(f"Converting {split_name} set...")
        
        for img_file, img_id in tqdm(data, desc=f"Processing {split_name}"):
            # Copy image
            img_output_dir = self.output_dir / split_name / 'images'
            img_output_path = img_output_dir / f"{img_id:08d}.jpg"
            shutil.copy2(img_file, img_output_path)
            
            # Get image dimensions
            with Image.open(img_file) as img:
                img_width, img_height = img.size
            
            # Create label file
            label_output_dir = self.output_dir / split_name / 'labels'
            label_output_path = label_output_dir / f"{img_id:08d}.txt"
            
            with open(label_output_path, 'w') as f:
                for ann in annotations[img_id]:
                    # Convert bbox to YOLO format
                    yolo_bbox = self.convert_bbox_to_yolo(
                        ann['bbox'], img_width, img_height
                    )
                    
                    # Write YOLO format line: class x_center y_center width height
                    line = f"{ann['class']} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                    f.write(line)
    
    def convert_test_data(self):
        """Convert test data to YOLO format."""
        
        print("Converting test set...")
        
        test_img_dir = self.source_dir / 'test' / 'img'
        test_output_dir = self.output_dir / 'test' / 'images'
        
        # Copy all test images
        for img_file in tqdm(test_img_dir.glob('*.jpg'), desc="Processing test"):
            shutil.copy2(img_file, test_output_dir / img_file.name)
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training."""
        
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': 1,  # number of classes
            'names': ['object']  # class names
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Created dataset configuration: {yaml_path}")
    
    def print_statistics(self):
        """Print dataset statistics."""
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / split / 'images'
            label_dir = self.output_dir / split / 'labels'
            
            n_images = len(list(img_dir.glob('*.jpg')))
            n_labels = len(list(label_dir.glob('*.txt'))) if label_dir.exists() else 0
            
            print(f"{split.upper()}: {n_images} images, {n_labels} labels")
            
            if split != 'test' and n_labels > 0:
                # Count total annotations
                total_objects = 0
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        total_objects += len(f.readlines())
                
                print(f"       {total_objects} total objects")
        
        print("="*50)
    
    def convert_all(self):
        """Convert all data to YOLO format."""
        
        print("Starting conversion to YOLO format...")
        
        # Convert training data
        self.convert_training_data()
        
        # Convert test data
        self.convert_test_data()
        
        # Create dataset configuration
        self.create_dataset_yaml()
        
        # Print statistics
        self.print_statistics()
        
        print(f"\nConversion completed! YOLO dataset available at: {self.output_dir}")


def main():
    """Main function."""
    
    # Configuration
    source_dir = "/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-1"
    output_dir = "/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-1/yolo_dataset"
    
    # Create converter
    converter = YOLOConverter(
        source_dir=source_dir,
        output_dir=output_dir,
        train_split=0.8,  # 80% for training
        val_split=0.2     # 20% for validation
    )
    
    # Convert dataset
    converter.convert_all()
    
    # Print usage instructions
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("1. Install YOLOv8:")
    print("   pip install ultralytics")
    print()
    print("2. Train YOLOv8 model:")
    print(f"   yolo train data={output_dir}/dataset.yaml model=yolov8n.pt epochs=100")
    print()
    print("3. For other YOLO versions, use the dataset.yaml file")
    print("   and adjust the training command accordingly.")
    print("="*60)


if __name__ == "__main__":
    main()
