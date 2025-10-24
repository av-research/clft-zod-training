#!/usr/bin/env python3
"""
ZOD Prediction Visualization Script

This script loads a trained model checkpoint and generates visualizations
of predicted segmentation masks overlaid on test images from the ZOD dataset.

Usage:
    python visualize_zod_predictions.py --checkpoint path/to/checkpoint_24.pth --output_dir ./visualizations

Features:
- Loads model from checkpoint
- Picks random test frames from ZOD dataset
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
import json

from clft.clft import CLFT
from utils.helpers import image_overlay, zod_anno_class_relabel, label_colors_list
from utils.metrics import zod_find_overlap_1


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


def get_random_test_frames(dataset_root, num_frames=5):
    """Get random test frames from ZOD dataset"""
    # Try different test split files
    test_files = [
        os.path.join(dataset_root, 'test_day_fair.txt'),
        os.path.join(dataset_root, 'test_night_fair.txt'),
        os.path.join(dataset_root, 'test_day_rain.txt'),
        os.path.join(dataset_root, 'test_night_rain.txt')
    ]

    all_frames = []
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                lines = f.read().splitlines()
                # Extract frame names from paths like "camera/frame_012236.png"
                for line in lines:
                    if '/' in line:
                        frame_name = line.split('/')[-1].replace('.png', '')
                    else:
                        frame_name = line.replace('.png', '')
                    all_frames.append(frame_name)

    if not all_frames:
        # Fallback to finding annotation files directly
        annotation_dir = os.path.join(dataset_root, 'annotation')
        if os.path.exists(annotation_dir):
            annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.png')]
            # Convert to frame names
            all_frames = [f.replace('.png', '') for f in annotation_files]
        else:
            raise FileNotFoundError(f"Could not find test frames. Checked {test_files} and {annotation_dir}")

    # Remove duplicates and select random frames
    all_frames = list(set(all_frames))
    test_frames = random.sample(all_frames, min(num_frames, len(all_frames)))

    return test_frames


def load_zod_frame(dataset_root, frame_name):
    """Load RGB image and annotation for a ZOD frame"""
    rgb_path = os.path.join(dataset_root, 'camera', f'{frame_name}.png')
    anno_path = os.path.join(dataset_root, 'annotation', f'{frame_name}.png')

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

    # RGB preprocessing
    rgb_transform = transforms.Compose([
        transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['Dataset']['transforms']['image_mean'],
            std=config['Dataset']['transforms']['image_std']
        )
    ])

    rgb_tensor = rgb_transform(rgb_pil)

    # For RGB-only mode, use RGB as both inputs
    lidar_tensor = rgb_tensor.clone()  # RGB-only fallback

    # Annotation preprocessing - keep as integers, don't normalize
    anno_transform = transforms.Compose([
        transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    # Convert PIL to numpy array first, then to tensor to preserve integer values
    anno_np = np.array(anno_pil)
    anno_tensor = torch.from_numpy(anno_np).long()
    
    # Apply ZOD annotation relabeling
    anno_tensor = zod_anno_class_relabel(anno_tensor)

    # Resize annotation to model size
    anno_tensor = anno_transform(anno_tensor)

    return rgb_tensor, lidar_tensor, anno_tensor


def create_segmentation_map_from_labels(labels):
    """Create segmentation map from class labels (H, W)"""
    labels = labels.astype(np.uint8)
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(len(label_colors_list)):
        if label_num < len(label_colors_list):
            idx = labels == label_num
            
            # Special handling for ignore class (1) - show as tiny dots
            if label_num == 1:  # ignore class
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
            
            red_map[idx] = label_colors_list[label_num][0]
            green_map[idx] = label_colors_list[label_num][1]
            blue_map[idx] = label_colors_list[label_num][2]

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
    # gt_mask is already resized to original dimensions in create_visualization
    # pred_labels comes from logits which are at model resolution
    # We need to resize pred_labels to match gt_mask dimensions
    
    h_gt, w_gt = gt_mask.shape
    h_pred, w_pred = pred_labels.shape
    
    # Resize pred_labels to match gt_mask if needed
    if (h_pred, w_pred) != (h_gt, w_gt):
        pred_labels_resized = cv2.resize(pred_labels.astype(np.uint8), (w_gt, h_gt), 
                                       interpolation=cv2.INTER_NEAREST)
    else:
        pred_labels_resized = pred_labels
    
    # Create comparison mask
    comparison_mask = np.zeros((h_gt, w_gt, 3), dtype=np.uint8)
    
    # Green for correct predictions, red for incorrect
    # Only evaluate pixels that are not background (class 0) in ground truth
    correct_mask = (gt_mask == pred_labels_resized) & (gt_mask > 0)
    incorrect_mask = (gt_mask != pred_labels_resized) & (gt_mask > 0)
    
    # Green (0, 255, 0) for correct - BGR format for OpenCV
    comparison_mask[correct_mask] = [0, 255, 0]
    
    # Red (0, 0, 255) for incorrect - BGR format for OpenCV
    comparison_mask[incorrect_mask] = [0, 0, 255]
    
    # Resize rgb_image to match comparison_mask size
    rgb_resized = cv2.resize(rgb_image, (w_gt, h_gt))
    
    # Apply transparent overlay on camera image
    alpha = 0.6  # Transparency level
    beta = 1 - alpha
    gamma = 0
    result = cv2.addWeighted(comparison_mask, alpha, rgb_resized, beta, gamma)
    
    return result


def create_visualization(rgb_original, pred_logits, gt_mask, frame_name, output_dir):
    """Create visualization with overlays"""
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to numpy
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    # Create segmentation maps for ground truth
    gt_seg = create_segmentation_map_from_labels(gt_mask)

    # Resize to original image dimensions
    h_orig, w_orig = rgb_original.size[1], rgb_original.size[0]  # PIL is (w, h)
    gt_seg_resized = cv2.resize(gt_seg, (w_orig, h_orig))

    # Convert PIL to CV2
    rgb_cv2 = cv2.cvtColor(np.array(rgb_original), cv2.COLOR_RGB2BGR)

    # Save results
    base_name = os.path.join(output_dir, frame_name)

    # Create and save ground truth overlays
    gt_overlay = image_overlay(rgb_cv2.copy(), gt_seg_resized)  # 40% opacity
    
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
        pred_seg = create_segmentation_map_from_labels(pred_labels)
        pred_seg_resized = cv2.resize(pred_seg, (w_orig, h_orig))

        # Create prediction overlays
        pred_overlay = image_overlay(rgb_cv2.copy(), pred_seg_resized)  # 40% opacity
        
        cv2.imwrite(f'{base_name}_prediction_overlay.png', pred_overlay)
        cv2.imwrite(f'{base_name}_prediction_mask.png', pred_seg_resized)

        # Create comparison overlay (green=correct, red=incorrect)
        comparison_overlay = create_comparison_overlay(gt_mask, pred_labels, rgb_cv2.copy())
        comparison_overlay_resized = cv2.resize(comparison_overlay, (w_orig, h_orig))
        cv2.imwrite(f'{base_name}_comparison_overlay.png', comparison_overlay_resized)

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
    parser = argparse.ArgumentParser(description='ZOD Ground Truth and Prediction Visualization')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (e.g., checkpoint_24.pth). If not provided, only ground truth will be visualized.')
    parser.add_argument('--config', type=str, default='config/zod/config_1.json',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./logs/zod_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_frames', type=int, default=1,
                        help='Number of random frames to visualize')
    parser.add_argument('--dataset_root', type=str, default='./zod_dataset',
                        help='Root directory of ZOD dataset')
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'lidar', 'cross_fusion'],
                        help='Fusion mode for CLFT model')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage instead of GPU')
    parser.add_argument('--frames', type=str, nargs='*', default=None,
                        help='Specific frame names to visualize (e.g., frame_004708). If not provided, selects random test frames.')

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
        test_frames = get_random_test_frames(args.dataset_root, args.num_frames)
        print(f"Selected {len(test_frames)} random test frames: {test_frames}")

    # Process each frame
    for frame_name in test_frames:
        try:
            print(f"\nProcessing frame: {frame_name}")

            # Load frame
            rgb_pil, anno_pil = load_zod_frame(args.dataset_root, frame_name)

            # Preprocess
            rgb_tensor, lidar_tensor, gt_tensor = preprocess_frame(rgb_pil, anno_pil, config)

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

                # Compute IoU for this frame
                n_classes = len(config['Dataset']['classes'])
                pred_batch = pred_logits.unsqueeze(0).cpu()  # [1, 6, 384, 384] on cpu
                gt_squeezed = gt_tensor.squeeze(0)  # [384,384] on cpu
                area_overlap, area_pred, area_label, area_union = zod_find_overlap_1(n_classes, pred_batch, gt_squeezed.unsqueeze(0))
                iou_per_class = area_overlap / area_union
                print(f"IoU for {frame_name}: Vehicle={iou_per_class[0]:.4f}, Sign={iou_per_class[1]:.4f}, Cyclist={iou_per_class[2]:.4f}, Pedestrian={iou_per_class[3]:.4f}")

            # Create visualizations
            create_visualization(rgb_pil, pred_logits, gt_tensor.cpu().squeeze(0),
                               frame_name, args.output_dir)

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