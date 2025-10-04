#!/usr/bin/env python3
"""
Script to verify class mappings and visualize segmentation on images.
Finds images containing pedestrian, cyclist, sign, and vehicle classes,
then overlays colored segmentation on the original image.
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse

def create_color_mapping():
    """Create color mapping for each class."""
    # Define colors for each class (RGB tuples)
    color_mapping = {
        0: (0, 0, 0),        # Background/Other - Black
        1: (0, 255, 0),      # Pedestrian - Green
        2: (255, 0, 0),      # Sign - Red
        3: (255, 255, 0),     # Cyclist - Yellow
    }
    return color_mapping

def check_classes_present(annotation_path, required_classes=[1, 2, 3, 4]):
    """Check if annotation contains all required classes."""
    try:
        img = Image.open(annotation_path)
        img_array = np.array(img)

        # Get unique pixel values (class IDs)
        unique_classes = set(np.unique(img_array))

        # Check if all required classes are present
        return all(cls in unique_classes for cls in required_classes)
    except Exception as e:
        print(f"Error checking {annotation_path}: {e}")
        return False

def create_colored_mask(annotation_array, color_mapping):
    """Create colored mask from annotation array."""
    height, width = annotation_array.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in color_mapping.items():
        mask = (annotation_array == class_id)
        colored_mask[mask] = color

    return colored_mask

def visualize_image_with_segmentation(rgb_path, annotation_path, color_mapping, output_path=None):
    """Visualize RGB image with colored segmentation overlay using PIL."""
    try:
        # Load RGB image
        rgb_img = Image.open(rgb_path)
        rgb_array = np.array(rgb_img)

        # Load annotation
        anno_img = Image.open(annotation_path)
        anno_array = np.array(anno_img)

        # Create colored mask
        colored_mask = create_colored_mask(anno_array, color_mapping)

        # Handle RGBA images (convert to RGB if necessary)
        if rgb_array.shape[-1] == 4:
            # Convert RGBA to RGB by removing alpha channel
            rgb_array = rgb_array[:, :, :3]

        # Create overlay (blend original image with colored mask)
        # Use alpha blending where mask is not background
        alpha = 0.5
        overlay = rgb_array.copy().astype(np.float32)

        # Apply mask where pixels are not background
        mask_pixels = (anno_array > 0)
        overlay[mask_pixels] = (1 - alpha) * rgb_array[mask_pixels] + alpha * colored_mask[mask_pixels]

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Convert overlay to PIL Image
        overlay_img = Image.fromarray(overlay)

        if output_path:
            overlay_img.save(output_path)
            print(f"Saved overlay image to: {output_path}")
        else:
            print("No output path provided, overlay not saved.")

        return True

    except Exception as e:
        print(f"Error visualizing {rgb_path} and {annotation_path}: {e}")
        return False

def find_image_with_all_classes(base_path, conditions=None):
    """Find an image that contains all required classes."""
    if conditions is None:
        conditions = [
            "day/not_rain",
            "day/rain",
            "night/not_rain",
            "night/rain"
        ]

    required_classes = [1, 2, 3]  # Pedestrian, Sign, Cyclist

    print("Searching for image containing all classes: Pedestrian, Sign, Cyclist")
    print("=" * 70)

    for condition in conditions:
        condition_path = os.path.join(base_path, "labeled", condition, "annotation")

        if not os.path.exists(condition_path):
            print(f"Warning: Path not found: {condition_path}")
            continue

        print(f"\nSearching in condition: {condition}")
        print("-" * 40)

        # Get all PNG files in this condition
        annotation_files = [f for f in os.listdir(condition_path) if f.endswith('.png')]
        annotation_files.sort()

        print(f"Checking {len(annotation_files)} annotation files...")

        for filename in annotation_files:
            annotation_path = os.path.join(condition_path, filename)

            if check_classes_present(annotation_path, required_classes):
                print(f"Found image with all classes: {filename}")
                return condition, filename

    print("No image found containing all required classes.")
    return None, None

def scan_all_classes(base_path, dataset_type='waymo', conditions=None):
    """Scan all annotations and report unique classes present."""
    if dataset_type == 'waymo':
        if conditions is None:
            conditions = [
                "day/not_rain",
                "day/rain",
                "night/not_rain",
                "night/rain"
            ]
        annotation_dirs = [os.path.join(base_path, "labeled", condition, "annotation") for condition in conditions]
    elif dataset_type == 'zod':
        annotation_dirs = [os.path.join(base_path, "annotation")]
    else:
        raise ValueError("dataset_type must be 'waymo' or 'zod'")

    all_unique_classes = set()
    class_counts = {}  # {class_id: total_pixels}

    print(f"Scanning all annotations for unique classes in {dataset_type} dataset...")
    print("=" * 70)

    total_files = 0
    for anno_dir in annotation_dirs:
        if not os.path.exists(anno_dir):
            print(f"Warning: Path not found: {anno_dir}")
            continue

        print(f"Scanning directory: {anno_dir}")
        
        annotation_files = [f for f in os.listdir(anno_dir) if f.endswith('.png')]
        total_files += len(annotation_files)

        for filename in annotation_files:
            annotation_path = os.path.join(anno_dir, filename)
            try:
                img = Image.open(annotation_path)
                img_array = np.array(img)
                unique_classes = np.unique(img_array)
                all_unique_classes.update(unique_classes)
                
                for cls in unique_classes:
                    if cls not in class_counts:
                        class_counts[cls] = 0
                    class_counts[cls] += np.sum(img_array == cls)
            except Exception as e:
                print(f"Error processing {annotation_path}: {e}")

    print(f"\nScan complete. Processed {total_files} annotation files.")
    print(f"Unique classes found: {sorted(all_unique_classes)}")
    print("\nClass distribution across all annotations:")
    class_names = {
        0: "Background",
        1: "Pedestrian", 
        2: "Sign",
        3: "Cyclist"
    }
    total_pixels = sum(class_counts.values())
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
        name = class_names.get(cls, f"Unknown ({cls})")
        print(f"  Class {cls} ({name}): {count:,} pixels ({percentage:.2f}%)")
    
    return all_unique_classes

def main():
    parser = argparse.ArgumentParser(description='Verify class mappings and visualize segmentation')
    parser.add_argument('--dataset_path', type=str, default='/home/tom/projects/clft-zod-training/waymo_dataset',
                        help='Path to the Waymo dataset directory')
    parser.add_argument('--output_dir', type=str, default='/home/tom/projects/clft-zod-training/output',
                        help='Directory to save visualization outputs')
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to create visualization (default: True)')
    parser.add_argument('--scan_classes', action='store_true',
                        help='Scan all annotations to check unique classes present')
    parser.add_argument('--dataset_type', type=str, default='waymo', choices=['waymo', 'zod'],
                        help='Type of dataset (waymo or zod)')

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create color mapping
    color_mapping = create_color_mapping()
    print("Color mapping:")
    class_names = {
        0: "Background/Other",
        1: "Pedestrian",
        2: "Sign",
        3: "Cyclist"
    }
    for class_id, color in color_mapping.items():
        print(f"  Class {class_id} ({class_names.get(class_id, 'Unknown')}): RGB{color}")

    if args.scan_classes:
        # Scan all classes in the dataset
        scan_all_classes(args.dataset_path, args.dataset_type)
        return

    # Find image with all classes
    condition, filename = find_image_with_all_classes(args.dataset_path)

    if condition and filename:
        # Construct paths
        base_name = filename.replace('.png', '')
        rgb_path = os.path.join(args.dataset_path, "labeled", condition, "camera", filename)
        annotation_path = os.path.join(args.dataset_path, "labeled", condition, "annotation", filename)
        output_path = os.path.join(args.output_dir, f"{base_name}_visualization.png")

        print(f"\nProcessing image: {filename}")
        print(f"RGB path: {rgb_path}")
        print(f"Annotation path: {annotation_path}")

        if args.visualize:
            success = visualize_image_with_segmentation(rgb_path, annotation_path, color_mapping, output_path)
            if success:
                print("Visualization completed successfully!")
            else:
                print("Failed to create visualization.")
        else:
            print("Visualization skipped (--visualize not set)")

        # Print class distribution for this image
        print(f"\nClass distribution for {filename}:")
        try:
            anno_img = Image.open(annotation_path)
            anno_array = np.array(anno_img)
            unique, counts = np.unique(anno_array, return_counts=True)
            total_pixels = anno_array.size

            for class_id, count in zip(unique, counts):
                percentage = (count / total_pixels * 100)
                class_name = class_names.get(class_id, f"Class {class_id}")
                color = color_mapping.get(class_id, "Unknown")
                print(f"  Class {class_id} ({class_name}): {count:,} pixels ({percentage:.2f}%) - Color: {color}")
        except Exception as e:
            print(f"Error analyzing class distribution: {e}")

    else:
        print("Could not find suitable image for verification.")

if __name__ == "__main__":
    main()