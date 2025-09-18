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

def visualize_test_images(model, config, checkpoint_path, test_images_dir="./test_images"):
    """
    Visualize predictions on test images in test_images directory.
    Expects files: test_img.png, test_lidar.pkl, test_anno.png
    """
    import pickle
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Get RGB mean/std for denormalization
    rgb_mean = np.array(config['rgb_mean'])
    rgb_std = np.array(config['rgb_std'])
    
    # Create output directory
    output_dir = "./output/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Class info from config
    CLASS_COLORS = {i: tuple(color) for i, color in enumerate(config['class_colors'])}
    CLASS_NAMES = config['class_names']
    
    # Find test image files
    test_files = []
    for file in os.listdir(test_images_dir):
        if file.endswith('_img.png'):
            base_name = file.replace('_img.png', '')
            img_path = os.path.join(test_images_dir, f'{base_name}_img.png')
            lidar_path = os.path.join(test_images_dir, f'{base_name}_lidar.pkl')
            anno_path = os.path.join(test_images_dir, f'{base_name}_anno.png')
            
            if os.path.exists(img_path) and os.path.exists(lidar_path) and os.path.exists(anno_path):
                test_files.append((base_name, img_path, lidar_path, anno_path))
    
    if not test_files:
        print(f"No complete test sets found in {test_images_dir}")
        return
    
    print(f"Found {len(test_files)} test image sets")
    
    with torch.no_grad():
        for base_name, img_path, lidar_path, anno_path in test_files:
            print(f"Processing {base_name}...")
            
            # Load RGB image
            rgb = Image.open(img_path).convert('RGB')
            rgb = rgb.resize((384, 384), Image.BILINEAR)
            rgb = transforms.ToTensor()(rgb)
            rgb = transforms.Normalize(rgb_mean, rgb_std)(rgb)
            rgb = rgb.unsqueeze(0).to(device)
            
            # Load LiDAR data
            with open(lidar_path, 'rb') as f:
                lidar_data = pickle.load(f)
            
            # Process LiDAR similar to dataset loading
            from dataset import open_lidar, get_unresized_lid_img_val
            points_set, camera_coord = open_lidar(lidar_path, w_ratio=4, h_ratio=4,
                                                lidar_mean=config['lidar_mean'],
                                                lidar_std=config['lidar_std'],
                                                camera_mask_value=config['camera_mask_value'])
            X, Y, Z = get_unresized_lid_img_val(384, 384, points_set, camera_coord)
            lidar = torch.cat((X, Y, Z), 0).unsqueeze(0).to(device)
            
            # Load annotation
            anno = Image.open(anno_path)
            anno = anno.resize((384, 384), Image.NEAREST)
            anno = torch.from_numpy(np.array(anno)).long().unsqueeze(0).to(device)
            
            # Run prediction
            pred = model(rgb, lidar, config['mode'])
            pred_mask = torch.argmax(pred, dim=1)
            
            # Visualize and save
            visualize_sample(rgb.squeeze(0), anno.squeeze(0), pred_mask.squeeze(0), 
                           base_name, output_dir, CLASS_COLORS, rgb_mean, rgb_std)
            
            print(f"Saved visualizations for {base_name}")

def visualize_sample(rgb, gt_mask, pred_mask, sample_name, save_dir, class_colors, rgb_mean, rgb_std):
    """Save visualizations for one sample."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Original RGB (denormalize)
    rgb_img = (rgb.cpu().permute(1, 2, 0).numpy() * rgb_std + rgb_mean) * 255
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_img).save(os.path.join(save_dir, f'{sample_name}_rgb.png'))
    
    # Ground truth mask
    gt_img = mask_to_image(gt_mask.cpu().numpy(), class_colors)
    Image.fromarray(gt_img).save(os.path.join(save_dir, f'{sample_name}_gt.png'))
    
    # Prediction mask
    pred_img = mask_to_image(pred_mask.cpu().numpy(), class_colors)
    Image.fromarray(pred_img).save(os.path.join(save_dir, f'{sample_name}_pred.png'))
    
    # Comparison
    comp_img = create_comparison(gt_mask.cpu().numpy(), pred_mask.cpu().numpy())
    Image.fromarray(comp_img).save(os.path.join(save_dir, f'{sample_name}_comparison.png'))
    
    print(f"  -> Saved: {sample_name}_{{rgb,gt,pred,comparison}}.png")

def main():
    parser = argparse.ArgumentParser(description='Visualize segmentation results')
    parser.add_argument('--dataset', type=str, default='waymo', choices=['zod', 'waymo'], help='Dataset to use (zod or waymo)')
    parser.add_argument('--test-images', action='store_true', help='Visualize test images instead of dataset')
    args = parser.parse_args()
    
    # Load config
    config_file = f'config_{args.dataset}.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Find the latest checkpoint
    checkpoint_path = config['checkpoint_path']
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        # Look for latest checkpoint in model_path
        model_dir = config['save_dir']
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split('_')[2]))
            checkpoint_path = os.path.join(model_dir, checkpoints[-1])
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            sys.exit("No checkpoint found. Please train a model first.")
    
    # Model
    model = ViTSegmentation(config['mode'], config['num_classes'])
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        
        # Handle compiled model checkpoints (remove _orig_mod. prefix)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        sys.exit(f"Checkpoint not found: {checkpoint_path}")
    
    if args.test_images:
        # Visualize test images
        visualize_test_images(model, config, checkpoint_path)
    else:
        # Original dataset visualization
        visualize_dataset(model, config, checkpoint_path, args.dataset)

def visualize_dataset(model, config, checkpoint_path, dataset_name):
    """Original dataset visualization function."""
    # Paths
    data_dir = config['data_dir']
    split_file = config['split_file']
    save_dir = config['save_dir']
    
    # Use test split for visualization
    test_config = config.copy()
    test_config['split_file'] = test_config['split_file'].replace('train_all.txt', 'test_day_fair.txt')
    
    # Dataset
    dataset = GenericDataset(test_config, training=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Class info from config
    CLASS_COLORS = {i: tuple(color) for i, color in enumerate(config['class_colors'])}
    CLASS_NAMES = config['class_names']
    
    output_dir = f"./output/visualizations_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize first 10 samples
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Limit to first 10 samples
                break
                
            rgb, lidar, anno = batch
            rgb, lidar, anno = rgb.to(device), lidar.to(device), anno.to(device)
            
            pred = model(rgb, lidar, config['mode'])
            pred_mask = torch.argmax(pred, dim=1).squeeze(0)
            
            sample_name = f"sample_{i:03d}"
            visualize_sample(rgb.squeeze(0), anno.squeeze(0), pred_mask, sample_name, 
                           output_dir, CLASS_COLORS, 
                           np.array(config['rgb_mean']), np.array(config['rgb_std']))
            print(f"Saved visualizations for sample {i+1}/10")

if __name__ == '__main__':
    main()
