#!/usr/bin/env python3
"""
Visualization Script for ViT Segmentation Model
Loads a checkpoint, generates prediction masks, and creates visualizations:
- Original ground truth masks
- Predicted masks
- Comparison showing differences/errors
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import sys
import argparse

from dataset import GenericDataset
from model import ViTSegmentation

def mask_to_image(mask, colors):
    """Convert class mask to RGB image."""
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in colors.items():
        img[mask == cls] = color
    return img

def create_comparison(gt_mask, pred_mask):
    """Create comparison image: green for correct, red for wrong."""
    h, w = gt_mask.shape
    comp = np.zeros((h, w, 3), dtype=np.uint8)
    
    correct = (gt_mask == pred_mask) & (gt_mask != 0)  # Correct non-background
    wrong = (gt_mask != pred_mask) & (gt_mask != 0)    # Wrong predictions
    
    comp[correct] = [0, 255, 0]  # Green for correct
    comp[wrong] = [255, 0, 0]    # Red for wrong
    comp[gt_mask == 0] = [0, 0, 0]  # Black for background
    
    return comp

def visualize_sample(rgb, gt_mask, pred_mask, sample_name, save_dir, class_colors):
    """Save visualizations for one sample."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract base name
    base_name = os.path.basename(sample_name).replace('.png', '')
    
    # Original RGB
    rgb_img = (rgb.cpu().permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_img).save(os.path.join(save_dir, f'{base_name}_rgb.png'))
    
    # Ground truth mask
    gt_img = mask_to_image(gt_mask.cpu().numpy(), class_colors)
    Image.fromarray(gt_img).save(os.path.join(save_dir, f'{base_name}_gt.png'))
    
    # Prediction mask
    pred_img = mask_to_image(pred_mask.cpu().numpy(), class_colors)
    Image.fromarray(pred_img).save(os.path.join(save_dir, f'{base_name}_pred.png'))
    
    # Comparison
    comp_img = create_comparison(gt_mask.cpu().numpy(), pred_mask.cpu().numpy())
    Image.fromarray(comp_img).save(os.path.join(save_dir, f'{base_name}_comparison.png'))

def main():
    parser = argparse.ArgumentParser(description='Visualize segmentation results')
    parser.add_argument('--dataset', type=str, default='zod', choices=['zod', 'waymo'], help='Dataset to use (zod or waymo)')
    args = parser.parse_args()
    
    # Load config
    config_file = f'config_{args.dataset}.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Paths
    data_dir = config['data_dir']
    split_file = config['split_file']
    checkpoint_path = config['checkpoint_path']
    save_dir = config['save_dir']
    
    # Dataset
    dataset = GenericDataset(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Model
    model = ViTSegmentation(config['mode'], config['num_classes'])
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        sys.exit("No checkpoint found. Please provide a valid checkpoint path in config.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Class info from config
    CLASS_COLORS = {i: tuple(color) for i, color in enumerate(config['class_colors'])}
    CLASS_NAMES = config['class_names']
    
    # Visualize
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            rgb, lidar, anno = batch
            rgb, lidar, anno = rgb.to(device), lidar.to(device), anno.to(device)
            
            pred = model(rgb, lidar)
            pred_mask = torch.argmax(pred, dim=1).squeeze(0)
            
            sample_name = dataset.samples[i]
            visualize_sample(rgb.squeeze(0), anno.squeeze(0), pred_mask, sample_name, save_dir, CLASS_COLORS)
            print(f"Saved visualizations for {sample_name} ({i+1}/{len(dataset)})")

if __name__ == '__main__':
    main()
