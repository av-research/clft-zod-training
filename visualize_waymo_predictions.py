#!/usr/bin/env python3
"""
Waymo Prediction Visualization Script

This script loads a trained model checkpoint and generates visualizations
of predicted segmentation masks overlaid on test images from the Waymo dataset.

Usage:
    python visualize_waymo_predictions.py --checkpoint path/to/checkpoint_36.pth --output_dir ./logs/waymo_visualizations

Features:
- Loads model from checkpoint
- Picks random test frames from Waymo dataset
- Generates segmentation predictions
- Creates overlay visualizations with ground truth and predictions
- Saves results to output directory
"""

import os
import cv2
import torch
import argparse
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import json

from clft.clft import CLFT
from utils.helpers import image_overlay, waymo_anno_class_relabel, waymo_anno_class_relabel_1, label_colors_list
from utils.metrics import find_overlap_1

# Waymo colors for relabeled classes (after waymo_anno_class_relabel_1)
# 0: background - black
# 1: cyclist - blue  
# 2: pedestrian - red
# 3: sign - yellow
# 4: ignore - gray
waymo_label_colors_list = [
    [0, 0, 0],        # 0: background - black
    [255, 0, 0],      # 1: cyclist - blue
    [0, 0, 255],      # 2: pedestrian - red
    [0, 255, 255],    # 3: sign - yellow
    [128, 128, 128]   # 4: ignore - gray
]


def load_model_from_checkpoint(checkpoint_path, config, use_cpu=False):
    """Load model from checkpoint"""
    device = torch.device('cpu' if use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load model architecture
    resize = config['Dataset']['transforms']['resize']
    model = CLFT(
        RGB_tensor_size=(3, resize, resize),
        XYZ_tensor_size=(3, resize, resize),
        patch_size=config['CLFT']['patch_size'],
        emb_dim=config['CLFT']['emb_dim'],
        resample_dim=config['CLFT']['resample_dim'],
        read=config['CLFT']['read'],
        hooks=config['CLFT']['hooks'],
        reassemble_s=config['CLFT']['reassembles'],
        nclasses=len(config['Dataset']['classes']),
        type=config['CLFT']['type'],
        model_timm=config['CLFT']['model_timm'],
    )

    # Load checkpoint on CPU first to avoid GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")

    return model, device


def get_random_frames(dataset_root, num_frames=5, use_training=False):
    """Get random frames from Waymo dataset (test or training)"""
    if use_training:
        split_file = os.path.join(dataset_root, 'splits_clft', 'train_all.txt')
        split_name = "training"
    else:
        # Use test split files (same as test.py)
        test_files = [
            os.path.join(dataset_root, 'splits_clft', 'test_day_fair.txt'),
            os.path.join(dataset_root, 'splits_clft', 'test_night_fair.txt'),
            os.path.join(dataset_root, 'splits_clft', 'test_day_rain.txt'),
            os.path.join(dataset_root, 'splits_clft', 'test_night_rain.txt')
        ]
        split_name = "test"
        split_file = test_files[0]  # Default to first test file for single file selection
    
    if use_training:
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                lines = f.read().splitlines()
                all_frames = [line.strip() for line in lines if line.strip()]
        else:
            raise FileNotFoundError(f"Training split file not found: {split_file}")
    else:
        # For test data, combine all test files
        all_frames = []
        for test_file in test_files:
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        if line.strip():
                            all_frames.append(line.strip())

    if not all_frames:
        raise FileNotFoundError(f"No frames found in {split_name} split")

    # Remove duplicates and select random frames
    all_frames = list(set(all_frames))
    selected_frames = random.sample(all_frames, min(num_frames, len(all_frames)))

    print(f"Selected {len(selected_frames)} random {split_name} frames")
    return selected_frames


def load_waymo_frame(dataset_root, frame_name):
    """Load RGB image and annotation for a Waymo frame"""
    if 'waymo' in dataset_root.lower():
        # Waymo: frame_name is full path like 'labeled/day/not_rain/camera/segment-xxx.png'
        rgb_path = os.path.join(dataset_root, frame_name)
        anno_path = rgb_path.replace('/camera/', '/annotation/')
    else:
        # Fallback
        rgb_path = os.path.join(dataset_root, frame_name)
        anno_path = rgb_path.replace('/camera/', '/annotation/')

    if not os.path.exists(rgb_path) or not os.path.exists(anno_path):
        raise FileNotFoundError(f"Frame {frame_name} not found. RGB: {rgb_path}, Anno: {anno_path}")

    # Load RGB image
    rgb_pil = Image.open(rgb_path).convert('RGB')

    # Load annotation
    anno_pil = Image.open(anno_path)

    return rgb_pil, anno_pil


def preprocess_frame(rgb_pil, anno_pil, config):
    """Preprocess RGB and annotation for model input"""
    resize = config['Dataset']['transforms']['resize']

    # RGB preprocessing - crop top half and resize
    w_orig, h_orig = rgb_pil.size  # PIL is (w, h)
    delta = int(h_orig/2)
    top_crop_rgb = TF.crop(rgb_pil, delta, 0, h_orig - delta, w_orig)
    
    rgb_transform = transforms.Compose([
        transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['Dataset']['transforms']['image_mean'],
            std=config['Dataset']['transforms']['image_std']
        )
    ])

    rgb_tensor = rgb_transform(top_crop_rgb)

    # The CLFT model uses lidar as input to the Vision Transformer backbone
    # In RGB mode, we still need to provide image-like input for the transformer
    # Use the RGB data as LiDAR input (since both are 3-channel images)
    lidar_tensor = rgb_tensor.clone()

    # Annotation preprocessing - crop top half and resize
    w_anno, h_anno = anno_pil.size
    delta_anno = int(h_anno/2)
    top_crop_anno = TF.crop(anno_pil, delta_anno, 0, h_anno - delta_anno, w_anno)
    
    anno_transform = transforms.Compose([
        transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    # Convert PIL to numpy array first, then to tensor to preserve integer values
    anno_np = np.array(top_crop_anno)
    anno_tensor_orig = torch.from_numpy(anno_np).long().unsqueeze(0)  # Add channel dimension for resize
    
    # Apply Waymo annotation relabeling for model input (same as Dataset class)
    anno_tensor = waymo_anno_class_relabel_1(anno_tensor_orig.squeeze(0))

    # Resize annotation to model size
    anno_tensor = anno_transform(anno_tensor.unsqueeze(0)).squeeze(0)
    anno_tensor_orig = anno_transform(anno_tensor_orig).squeeze(0)

    return rgb_tensor, lidar_tensor, anno_tensor, anno_tensor_orig


def create_segmentation_map_from_labels(labels, colors_list):
    """Create segmentation map from class labels (H, W)"""
    labels = labels.astype(np.uint8)
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(len(colors_list)):
        if label_num < len(colors_list):
            idx = labels == label_num
            
            # Special handling for ignore class (4 for Waymo) - show as tiny dots
            ignore_class = 4  # Waymo ignore at 4
            if label_num == ignore_class:  # ignore class
                # Only show ~5% of ignore pixels as dots to avoid cluttering
                ignore_pixels = np.where(idx)
                if len(ignore_pixels[0]) > 0:
                    # Randomly select 5% of ignore pixels
                    num_dots = max(1, len(ignore_pixels[0]) // 20)  # 5% = 1/20
                    selected_indices = np.random.choice(len(ignore_pixels[0]), 
                                                      size=num_dots, replace=False)
                    # Create sparse mask for dots
                    sparse_idx = (ignore_pixels[0][selected_indices], 
                                ignore_pixels[1][selected_indices])
                    idx = np.zeros_like(idx)
                    idx[sparse_idx] = True
            
            red_map[idx] = colors_list[label_num][0]
            green_map[idx] = colors_list[label_num][1]
            blue_map[idx] = colors_list[label_num][2]

    segmented_image = np.stack([blue_map, green_map, red_map], axis=2)  # CV2 is BGR
    return segmented_image


def create_clear_overlay(image, segmented_image, alpha=0.7):
    """Create overlay with custom alpha transparency"""
    beta = 1 - alpha  # alpha + beta should equal 1
    gamma = 0  # scalar added to each sum
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image


def create_comparison_overlay(gt_mask, pred_labels, rgb_image):
    """Create comparison overlay: green for correct, red for incorrect predictions"""
    # gt_mask and pred_labels are already at model resolution (384x384)
    # rgb_image is already cropped to (160, 480) - height x width
    
    h_gt, w_gt = gt_mask.shape
    h_pred, w_pred = pred_labels.shape
    
    # Resize pred_labels to match gt_mask if needed (should already match)
    if (h_pred, w_pred) != (h_gt, w_gt):
        pred_labels_resized = cv2.resize(pred_labels.astype(np.uint8), (w_gt, h_gt), 
                                       interpolation=cv2.INTER_NEAREST)
    else:
        pred_labels_resized = pred_labels
    
    # Create comparison mask at model resolution
    comparison_mask = np.zeros((h_gt, w_gt, 3), dtype=np.uint8)
    
    # Green for correct predictions, red for incorrect
    # Only evaluate pixels that are not background (class 0) in ground truth
    correct_mask = (gt_mask == pred_labels_resized) & (gt_mask > 0)
    incorrect_mask = (gt_mask != pred_labels_resized) & (gt_mask > 0)
    
    # Green (0, 255, 0) for correct - BGR format for OpenCV
    comparison_mask[correct_mask] = [0, 255, 0]
    
    # Red (0, 0, 255) for incorrect - BGR format for OpenCV
    comparison_mask[incorrect_mask] = [0, 0, 255]
    
    # Resize comparison to match cropped image dimensions (160, 480)
    comparison_resized = cv2.resize(comparison_mask, (480, 160), interpolation=cv2.INTER_NEAREST)
    
    # rgb_image is already the correct size, no need to resize
    rgb_correct_size = rgb_image
    
    # Apply transparent overlay on cropped camera image
    alpha = 0.6  # Transparency level
    beta = 1 - alpha
    gamma = 0
    result = cv2.addWeighted(comparison_resized, alpha, rgb_correct_size, beta, gamma)
    
    return result


def create_visualization(rgb_original, pred_logits, gt_mask, frame_name, output_dir, config):
    """Create visualization with overlays"""
    base_name = os.path.join(output_dir, frame_name.replace('.png', ''))
    os.makedirs(os.path.dirname(base_name), exist_ok=True)

    # Convert tensors to numpy
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    # Choose color list for Waymo
    colors_list = waymo_label_colors_list

    # Create segmentation maps for ground truth
    gt_seg = create_segmentation_map_from_labels(gt_mask, colors_list)

    # Resize to cropped image dimensions (top half: 480x160 for Waymo)
    gt_seg_resized = cv2.resize(gt_seg, (480, 160))

    # Load original image and crop top half for overlay
    rgb_cv2 = cv2.imread(os.path.join('./waymo_dataset', frame_name))
    rgb_cv2_top = rgb_cv2[160:320, 0:480]  # Crop top half

    # Save results
    gt_overlay = image_overlay(rgb_cv2_top.copy(), gt_seg_resized)  # 40% opacity
    
    cv2.imwrite(f'{base_name}_ground_truth_overlay.png', gt_overlay)
    cv2.imwrite(f'{base_name}_ground_truth_mask.png', gt_seg_resized)

    # Only create prediction visualizations if predictions are available
    if pred_logits is not None:
        # Convert tensors to numpy
        if isinstance(pred_logits, torch.Tensor):
            pred_logits = pred_logits.detach().cpu().numpy()

        # Get predicted class labels from logits
        pred_labels = np.argmax(pred_logits, axis=0).astype(np.uint8)
        
        # Create segmentation maps for predictions
        pred_seg = create_segmentation_map_from_labels(pred_labels, colors_list)
        pred_seg_resized = cv2.resize(pred_seg, (480, 160))
        pred_overlay = image_overlay(rgb_cv2_top.copy(), pred_seg_resized)  # 40% opacity
        
        cv2.imwrite(f'{base_name}_prediction_overlay.png', pred_overlay)
        cv2.imwrite(f'{base_name}_prediction_mask.png', pred_seg_resized)

        # Create comparison overlay (green=correct, red=incorrect)
        comparison_overlay = create_comparison_overlay(gt_mask, pred_labels, rgb_cv2_top.copy())
        cv2.imwrite(f'{base_name}_comparison_overlay.png', comparison_overlay)

    print(f"Saved visualizations for {frame_name} to {output_dir}")


def get_model_second_input(rgb_tensor, lidar_tensor, modality):
    """
    Get the correct second input (rgb or lidar) based on modality.
    
    Modality modes:
    - 'rgb': Use RGB for both inputs (RGB-only mode)
    - 'lidar': Use LiDAR as second input (LiDAR-only mode)
    - 'cross_fusion': Use LiDAR as second input (cross-modal fusion)
    
    Args:
        rgb_tensor: RGB image tensor
        lidar_tensor: LiDAR image tensor
        modality: String specifying the modality mode
        
    Returns:
        Tensor to use as second input to the model
    """
    if modality == 'rgb':
        # RGB-only mode: use RGB for both inputs
        return rgb_tensor
    else:
        # LiDAR or cross_fusion modes: use LiDAR as second input
        return lidar_tensor


def main():
    parser = argparse.ArgumentParser(description='Waymo Ground Truth and Prediction Visualization')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (e.g., checkpoint_36.pth). If not provided, only ground truth will be visualized.')
    parser.add_argument('--config', type=str, default='config/waymo/config_7.json',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./logs/waymo_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_frames', type=int, default=1,
                        help='Number of random frames to visualize')
    parser.add_argument('--dataset_root', type=str, default='./waymo_dataset',
                        help='Root directory of Waymo dataset')
    parser.add_argument('--mode', type=str, default='cross_fusion', choices=['rgb', 'lidar', 'cross_fusion'],
                        help='Fusion mode for CLFT model')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage instead of GPU')
    parser.add_argument('--frames', type=str, nargs='*', default=None,
                        help='Specific frame names to visualize (e.g., labeled/day/not_rain/camera/segment-xxx.png). If not provided, selects random frames.')
    parser.add_argument('--use_training', action='store_true',
                        help='Use training data instead of test data')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load model if checkpoint is provided
    model = None
    device = None
    if args.checkpoint:
        model, device = load_model_from_checkpoint(args.checkpoint, config, args.cpu)
        print(f"Model loaded from: {args.checkpoint}")
    else:
        print("No checkpoint provided - generating ground truth visualizations only")

    # Get frames to process
    if args.frames:
        test_frames = args.frames
        print(f"Using specified frames: {test_frames}")
    else:
        test_frames = get_random_frames(args.dataset_root, args.num_frames, args.use_training)
        print(f"Selected {len(test_frames)} random {'training' if args.use_training else 'test'} frames: {test_frames[:3]}...")

    # Process each frame
    for frame_name in test_frames:
        try:
            print(f"\nProcessing frame: {frame_name}")

            # Load frame
            rgb_pil, anno_pil = load_waymo_frame(args.dataset_root, frame_name)

            # Preprocess
            rgb_tensor, lidar_tensor, gt_tensor, gt_original = preprocess_frame(rgb_pil, anno_pil, config)

            # Run inference if model is available
            pred_logits = None
            if model is not None:
                # Add batch dimension
                rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
                lidar_tensor = lidar_tensor.unsqueeze(0).to(device)

                # Get correct input based on modality
                second_input = get_model_second_input(rgb_tensor, lidar_tensor, args.mode)

                # Run inference
                with torch.no_grad():
                    _, pred_logits = model(rgb_tensor, second_input, args.mode)
                
                pred_logits = pred_logits.squeeze(0)
                pred_logits = pred_logits.cpu()

                # Compute IoU for this frame
                n_classes = len(config['Dataset']['classes'])
                pred_batch = pred_logits.unsqueeze(0)  # [1, n_classes, 384, 384] on cpu
                gt_squeezed = gt_tensor.squeeze(0)  # [384,384] on cpu - use relabeled annotations for IoU calculation
                area_overlap, area_pred, area_label, area_union = find_overlap_1(n_classes, pred_batch, gt_squeezed.unsqueeze(0))
                iou_per_class = area_overlap / area_union
                class_names = ['cyclist', 'pedestrian', 'sign']
                print(f"IoU for {frame_name}: " + ", ".join([f"{name}={iou:.4f}" for name, iou in zip(class_names, iou_per_class)]))

                # Debug: print unique classes in predictions and GT
                pred_labels = np.argmax(pred_logits.numpy(), axis=0).astype(np.uint8)
                gt_mask_np = gt_squeezed.numpy()
                print(f"Unique in predictions: {np.unique(pred_labels)}")
                print(f"Unique in GT: {np.unique(gt_mask_np)}")

            # Create visualizations
            create_visualization(rgb_pil, pred_logits, gt_tensor.cpu().squeeze(0),
                               frame_name, args.output_dir, config)

        except Exception as e:
            print(f"Error processing frame {frame_name}: {e}")
            continue

    print(f"\nVisualization complete! Results saved to {args.output_dir}")
    if model is None:
        print("Generated ground truth visualizations only (no predictions)")
    else:
        print("Generated both ground truth and prediction visualizations")


if __name__ == '__main__':
    main()