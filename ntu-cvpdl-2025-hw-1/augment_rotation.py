#!/usr/bin/env python3
"""
Data Augmentation Script - Rotation
Rotates images and adjusts YOLO format labels accordingly
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm
import argparse


def rotate_image(image, angle):
    """
    Rotate image by given angle (in degrees)
    Returns rotated image and rotation matrix
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions to fit rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=(0, 0, 0))
    
    return rotated, M, (new_w, new_h)


def rotate_bbox(bbox, M, orig_size, new_size):
    """
    Rotate YOLO format bounding box
    bbox: [x_center, y_center, width, height] (normalized 0-1)
    Returns: rotated bbox in same format
    """
    orig_w, orig_h = orig_size
    new_w, new_h = new_size
    
    # Convert normalized to absolute coordinates
    x_center = bbox[0] * orig_w
    y_center = bbox[1] * orig_h
    width = bbox[2] * orig_w
    height = bbox[3] * orig_h
    
    # Get 4 corners of the bounding box
    corners = np.array([
        [x_center - width/2, y_center - height/2, 1],  # top-left
        [x_center + width/2, y_center - height/2, 1],  # top-right
        [x_center + width/2, y_center + height/2, 1],  # bottom-right
        [x_center - width/2, y_center + height/2, 1],  # bottom-left
    ]).T
    
    # Apply rotation to corners
    rotated_corners = M @ corners
    rotated_corners = rotated_corners.T
    
    # Get new bounding box from rotated corners
    x_coords = rotated_corners[:, 0]
    y_coords = rotated_corners[:, 1]
    
    new_x_min = np.min(x_coords)
    new_x_max = np.max(x_coords)
    new_y_min = np.min(y_coords)
    new_y_max = np.max(y_coords)
    
    # Convert back to center format
    new_x_center = (new_x_min + new_x_max) / 2
    new_y_center = (new_y_min + new_y_max) / 2
    new_width = new_x_max - new_x_min
    new_height = new_y_max - new_y_min
    
    # Clip to image bounds
    new_x_center = np.clip(new_x_center, 0, new_w)
    new_y_center = np.clip(new_y_center, 0, new_h)
    new_width = np.clip(new_width, 0, new_w)
    new_height = np.clip(new_height, 0, new_h)
    
    # Normalize
    new_x_center /= new_w
    new_y_center /= new_h
    new_width /= new_w
    new_height /= new_h
    
    return [new_x_center, new_y_center, new_width, new_height]


def augment_dataset(dataset_path, angles=[90, 180, 270], sets=['train', 'val']):
    """
    Augment dataset by rotating images and labels
    
    Args:
        dataset_path: Path to YOLO dataset root
        angles: List of rotation angles to apply
        sets: List of dataset splits to augment ['train', 'val', 'test']
    """
    dataset_path = Path(dataset_path)
    
    for split in sets:
        img_dir = dataset_path / split / 'images'
        label_dir = dataset_path / split / 'labels'
        
        if not img_dir.exists():
            print(f"Warning: {img_dir} does not exist, skipping...")
            continue
            
        if not label_dir.exists():
            print(f"Warning: {label_dir} does not exist, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Augmenting {split} set with rotations: {angles}")
        print(f"{'='*60}")
        
        # Get all images
        image_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        
        augmented_count = 0
        
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            orig_h, orig_w = img.shape[:2]
            
            # Read corresponding label
            label_path = label_dir / (img_path.stem + '.txt')
            bboxes = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]
                            bboxes.append((class_id, bbox))
            
            # Apply each rotation angle
            for angle in angles:
                # Rotate image
                rotated_img, M, (new_w, new_h) = rotate_image(img, angle)
                
                # Save rotated image
                rotated_img_name = f"{img_path.stem}_rot{angle}{img_path.suffix}"
                rotated_img_path = img_dir / rotated_img_name
                cv2.imwrite(str(rotated_img_path), rotated_img)
                
                # Rotate bounding boxes
                rotated_bboxes = []
                for class_id, bbox in bboxes:
                    rotated_bbox = rotate_bbox(bbox, M, (orig_w, orig_h), (new_w, new_h))
                    
                    # Filter out invalid boxes (too small or out of bounds)
                    if (rotated_bbox[2] > 0.01 and rotated_bbox[3] > 0.01 and
                        rotated_bbox[0] > 0 and rotated_bbox[0] < 1 and
                        rotated_bbox[1] > 0 and rotated_bbox[1] < 1):
                        rotated_bboxes.append((class_id, rotated_bbox))
                
                # Save rotated label
                rotated_label_path = label_dir / f"{img_path.stem}_rot{angle}.txt"
                with open(rotated_label_path, 'w') as f:
                    for class_id, bbox in rotated_bboxes:
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                augmented_count += 1
        
        original_count = len(image_files)
        total_count = original_count + augmented_count
        print(f"\n{split} set:")
        print(f"  Original images: {original_count}")
        print(f"  Augmented images: {augmented_count}")
        print(f"  Total images: {total_count}")


def main():
    parser = argparse.ArgumentParser(description='Augment YOLO dataset with rotations')
    parser.add_argument('--dataset', type=str, default='yolo_dataset',
                      help='Path to YOLO dataset root directory')
    parser.add_argument('--angles', type=int, nargs='+', default=[90, 180, 270],
                      help='Rotation angles to apply (default: 90 180 270)')
    parser.add_argument('--sets', type=str, nargs='+', default=['train', 'val'],
                      help='Dataset splits to augment (default: train val)')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist!")
        return
    
    print("="*60)
    print("YOLO Dataset Rotation Augmentation")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Rotation angles: {args.angles}")
    print(f"Dataset splits: {args.sets}")
    print("="*60)
    
    # Perform augmentation
    augment_dataset(dataset_path, angles=args.angles, sets=args.sets)
    
    print("\n" + "="*60)
    print("Augmentation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
