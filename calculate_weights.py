#!/usr/bin/env python3
"""
Weight Calculator Script
Reads split_file, counts all labels in annotations, and calculates class weights.
"""

import os
import argparse
import json
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import torch

# Import relabel functions
from utils.helpers import waymo_anno_class_relabel, zod_anno_class_relabel

def calculate_class_weights(dataset_name, config):
    """
    Calculate class weights based on pixel frequencies in the dataset.
    """
    data_dir = config['data_dir']
    split_file = config['split_file']
    num_classes = config['num_classes']
    class_names = config.get('class_names', [f'class_{i}' for i in range(num_classes)])
    
    # Read split file
    with open(split_file, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(samples)} samples in {split_file}")
    
    # Count pixels per class
    class_counts = defaultdict(int)
    total_pixels = 0
    
    for sample in tqdm(samples, desc="Processing samples"):
        # Construct annotation path
        anno_path = os.path.join(data_dir, sample.replace('/camera/', '/annotation/'))
        if not os.path.exists(anno_path):
            print(f"Warning: {anno_path} not found, skipping")
            continue
        
        # Load and relabel annotation
        anno_img = Image.open(anno_path)
        if dataset_name == 'waymo':
            # For Waymo, check raw annotation first
            anno = np.array(anno_img)
            print(f"Raw unique values in Waymo annotation: {np.unique(anno)}")
            # anno = waymo_anno_class_relabel(anno_img)
            anno = torch.from_numpy(anno).unsqueeze(0).long()
        elif dataset_name == 'zod':
            anno = zod_anno_class_relabel(anno_img)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Convert to numpy
        anno_np = np.array(anno)
        total_pixels += anno_np.size
        
        # Count per class
        unique, counts = np.unique(anno_np, return_counts=True)
        for cls, count in zip(unique, counts):
            if 0 <= cls < num_classes:
                class_counts[cls] += count
    
    print(f"Total pixels processed: {total_pixels}")
    if total_pixels == 0:
        print("No valid samples found. Check split_file and data paths.")
        return [1.0] * num_classes
    
    print("Class counts:")
    for cls in range(num_classes):
        count = class_counts.get(cls, 0)
        name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
        print(f"  {name} ({cls}): {count} pixels ({count/total_pixels*100:.2f}%)")
    
    # Calculate weights (inverse frequency)
    raw_weights = []
    for cls in range(num_classes):
        count = class_counts.get(cls, 1)  # Avoid division by zero
        weight = total_pixels / (num_classes * count)
        raw_weights.append(weight)
    
    # Cap maximum weight to prevent numerical issues
    max_weight = 1000.0
    raw_weights = [min(w, max_weight) for w in raw_weights]
    
    # Normalize so minimum weight is 1
    min_weight = min(raw_weights)
    normalized_weights = [w / min_weight for w in raw_weights]
    
    # Round to reasonable precision and convert to float
    raw_weights = [round(float(w), 4) for w in raw_weights]
    normalized_weights = [round(float(w), 4) for w in normalized_weights]
    
    print(f"Raw inverse frequency weights: {raw_weights}")
    print(f"Normalized weights (min=1): {normalized_weights}")
    
    return normalized_weights

def main():
    parser = argparse.ArgumentParser(description='Calculate class weights for dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=['zod', 'waymo'], help='Dataset to calculate weights for')
    args = parser.parse_args()
    
    # Load config
    config_file = f'config_{args.dataset}.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    weights = calculate_class_weights(args.dataset, config)
    
    print(f"\nRecommended class_weights for config_{args.dataset}.json:")
    print(f'  "class_weights": {weights}')

if __name__ == '__main__':
    main()
