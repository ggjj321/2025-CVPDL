#!/usr/bin/env python3
"""
Generate submission file from YOLO predictions
Converts YOLO prediction format to competition submission format
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


def convert_yolo_to_submission(pred_label_dir, test_img_dir, output_csv):
    """
    Convert YOLO predictions to submission format
    
    Args:
        pred_label_dir: Directory containing YOLO prediction txt files
        test_img_dir: Directory containing test images
        output_csv: Output CSV file path
    """
    pred_label_dir = Path(pred_label_dir)
    test_img_dir = Path(test_img_dir)
    
    # Get all test images
    test_images = sorted(list(test_img_dir.glob('*.png')) + list(test_img_dir.glob('*.jpg')))
    
    print(f"Processing {len(test_images)} test images...")
    
    submission_data = []
    
    for img_path in tqdm(test_images):
        # Extract image ID from filename (e.g., img0001.png -> 1)
        img_id = int(img_path.stem.replace('img', ''))
        
        # Read image dimensions
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        # Read prediction file
        pred_file = pred_label_dir / (img_path.stem + '.txt')
        
        prediction_strings = []
        
        if pred_file.exists():
            with open(pred_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    
                    # YOLO format: class x_center y_center width height confidence
                    class_id = int(parts[0])
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])
                    confidence = float(parts[5])
                    
                    # Convert normalized coordinates to absolute
                    x_center = x_center_norm * img_width
                    y_center = y_center_norm * img_height
                    width = width_norm * img_width
                    height = height_norm * img_height
                    
                    # Convert center format to top-left format
                    bb_left = x_center - width / 2
                    bb_top = y_center - height / 2
                    
                    # Format: <conf> <bb_left> <bb_top> <bb_width> <bb_height> <class>
                    pred_str = f"{confidence:.6f} {bb_left:.2f} {bb_top:.2f} {width:.2f} {height:.2f} {class_id}"
                    prediction_strings.append(pred_str)
        
        # Join all predictions with space
        prediction_string = ' '.join(prediction_strings)
        
        submission_data.append({
            'Image_ID': img_id,
            'PredictionString': prediction_string
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(submission_data)
    df = df.sort_values('Image_ID')
    df.to_csv(output_csv, index=False)
    
    print(f"\nSubmission file saved: {output_csv}")
    print(f"Total images: {len(df)}")
    print(f"Images with detections: {(df['PredictionString'] != '').sum()}")
    print(f"Empty predictions: {(df['PredictionString'] == '').sum()}")
    
    # Show sample
    print("\nSample predictions:")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        pred_str = row['PredictionString']
        if len(pred_str) > 100:
            pred_str = pred_str[:100] + "..."
        print(f"  Image {row['Image_ID']}: {pred_str}")


def main():
    # Paths
    base_dir = Path('/home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-2')
    
    # Find latest prediction directory
    runs_dir = base_dir / 'runs' / 'detect'
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        print("Please run YOLO prediction first!")
        return
    
    # Get all predict directories
    predict_dirs = sorted([d for d in runs_dir.glob('predict*') if d.is_dir()],
                          key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not predict_dirs:
        print(f"Error: No prediction directories found in {runs_dir}")
        print("Please run YOLO prediction first!")
        return
    
    # Use the latest prediction directory
    latest_pred_dir = predict_dirs[0]
    pred_label_dir = latest_pred_dir / 'labels'
    
    if not pred_label_dir.exists():
        print(f"Error: Labels directory not found: {pred_label_dir}")
        return
    
    print(f"Using prediction directory: {latest_pred_dir}")
    
    # Test images directory
    test_img_dir = base_dir / 'CVPDL_hw2' / 'CVPDL_hw2' / 'test'
    
    if not test_img_dir.exists():
        print(f"Error: Test directory not found: {test_img_dir}")
        return
    
    # Output CSV
    output_csv = base_dir / 'submission.csv'
    
    print("="*60)
    print("YOLO to Submission Converter")
    print("="*60)
    print(f"Prediction labels: {pred_label_dir}")
    print(f"Test images: {test_img_dir}")
    print(f"Output CSV: {output_csv}")
    print("="*60)
    
    # Convert predictions
    convert_yolo_to_submission(pred_label_dir, test_img_dir, output_csv)
    
    print("\n" + "="*60)
    print("Conversion completed!")
    print("="*60)


if __name__ == "__main__":
    main()
